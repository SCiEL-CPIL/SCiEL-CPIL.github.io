import time
import numpy as np
import random
from itertools import combinations, product

# EVOLVE-BLOCK-START

def _int_to_digits(x, q, n, powers):
    """Convert integer index to list of digits."""
    digits = []
    for i in range(n):
        d = (x // powers[i]) % q
        digits.append(d)
    return digits

def _digits_to_int(digits, q, n, powers):
    """Convert list of digits to integer index."""
    x = 0
    for i, d in enumerate(digits):
        x += d * powers[i]
    return x

def _generate_linear_code(q, n, k, deadline):
    """
    Generate a random linear code of dimension k over F_q.
    Returns a list of integer indices representing the codewords.
    """
    if k > n:
        k = n
        
    # Generate a random generator matrix G of size k x n
    G = np.random.randint(0, q, size=(k, n))
    
    # Ensure rows are linearly independent? 
    # For a covering code, we just need a set of distinct codewords.
    # If rows are dependent, we might have duplicates, but we can filter them.
    # Actually, a linear code of dimension k has exactly q^k codewords.
    # If G is not full rank, the code is smaller. 
    # Let's just generate all linear combinations of the rows.
    
    codewords = set()
    # Iterate over all 0..q^k - 1
    total_combos = q ** k
    # This might be slow if k is large, but for n=8, k<=8, q=3, max 3^8=6561 which is fine.
    # However, we want to start with a reasonable size. 
    # A (8, k) code has 3^k codewords.
    # K_3(8,1) is around 400-500.
    # 3^5 = 243, 3^6 = 729.
    # So k=5 or k=6 is a good starting point.
    
    # To speed up, we can just generate the basis and expand.
    basis = [G[i] for i in range(k)]
    
    # Precompute powers for digit conversion if needed, but we can work in vector space directly?
    # The evaluator wants indices. Let's generate vectors and convert.
    # Or better, generate indices directly?
    # Vector arithmetic is easier.
    
    # Generate all codewords
    all_cw = []
    # Use itertools.product to get coefficients
    for coeffs in product(range(q), repeat=k):
        cw_vec = np.zeros(n, dtype=int)
        for i in range(k):
            if coeffs[i] != 0:
                cw_vec += coeffs[i] * basis[i]
        # Modulo q
        cw_vec = cw_vec % q
        
        # Convert to index
        idx = 0
        for j in range(n):
            idx += cw_vec[j] * (q ** j)
        all_cw.append(idx)
        
    # Remove duplicates (should be none if G is full rank, but safe)
    unique_cw = list(set(all_cw))
    
    return unique_cw

def _precompute_balls(q, n, radius, deadline):
    """
    Precompute for each word x the list of words within Hamming distance <= radius.
    Returns a list of lists.
    """
    total = q ** n
    if radius == 1:
        balls = []
        powers = [1] * n
        for i in range(1, n):
            powers[i] = powers[i-1] * q
        
        for x in range(total):
            ball = [x]
            for i in range(n):
                old_digit = (x // powers[i]) % q
                for new_digit in range(q):
                    if new_digit != old_digit:
                        y = x + (new_digit - old_digit) * powers[i]
                        ball.append(y)
            balls.append(ball)
            if time.time() > deadline - 10:
                break
        return balls
    else:
        raise NotImplementedError("Only radius 1 is optimized")

def _check_coverage(selected, balls, total, deadline):
    """
    Check if the selected codewords cover the entire space.
    Returns True if covered, False otherwise.
    """
    covered = np.zeros(total, dtype=bool)
    for c in selected:
        for y in balls[c]:
            covered[y] = True
            
    # Check if all are covered
    # This is O(total) which is 6561, very fast.
    return np.all(covered)

def _get_uncovered(selected, balls, total):
    """
    Returns a boolean array where True indicates uncovered words.
    """
    covered = np.zeros(total, dtype=bool)
    for c in selected:
        for y in balls[c]:
            covered[y] = True
    return ~covered

def _vns_search(initial_set, balls, inv, total, q, n, deadline):
    """
    Variable Neighborhood Search (VNS) to minimize the size of the covering set.
    Neighborhoods:
    N1: Remove one codeword if possible.
    N2: Swap one codeword with another (if it allows removal of a different one? No, standard swap: replace c1 with c2, then try to remove redundant).
    Actually, simpler VNS for Set Cover:
    1. Try to remove any redundant codeword (1-move).
    2. If stuck, try swapping one codeword with a random one from the universe (2-move).
       If the new set is a valid cover, accept it (even if size is same, we might be able to remove later).
       Actually, we want to minimize size. 
       Standard approach: 
       - Phase 1: Remove redundant.
       - Phase 2: If no redundant, pick a random codeword c_out, pick a random c_in from universe \ current.
         New set = (Current - {c_out}) U {c_in}.
         If New set covers space, accept. Then go back to Phase 1.
    """
    current_set = list(initial_set)
    current_size = len(current_set)
    
    # Precompute cover counts for current set
    # cover_count[y] = number of selected codewords covering y
    cover_count = np.zeros(total, dtype=np.int32)
    for c in current_set:
        for y in balls[c]:
            cover_count[y] += 1
    
    # Helper to check if a codeword is redundant given current cover_count
    # It is redundant if min(cover_count[ball[c]]) >= 2
    
    # Helper to try removing all possible redundant codewords
    def try_remove_redundant():
        nonlocal current_set, current_size, cover_count
        improved = True
        while improved:
            improved = False
            # Check each codeword in current_set
            # To be efficient, we can iterate and check
            # Since set size is ~400-500, and ball size is 25, this is fast.
            candidates_to_remove = []
            for c in current_set:
                # Check if c is redundant
                # min count in ball[c] must be >= 2
                min_cnt = np.min(cover_count[balls[c]])
                if min_cnt >= 2:
                    candidates_to_remove.append(c)
                    
            if candidates_to_remove:
                # Remove one (randomly to avoid bias)
                c_to_remove = random.choice(candidates_to_remove)
                # Update cover_count
                for y in balls[c_to_remove]:
                    cover_count[y] -= 1
                current_set.remove(c_to_remove)
                current_size -= 1
                improved = True
        return current_size

    # Main VNS Loop
    last_improvement_time = time.time()
    stagnation_counter = 0
    
    while time.time() < deadline - 5:
        # Phase 1: Remove redundant
        new_size = try_remove_redundant()
        if new_size < current_size:
            current_size = new_size
            last_improvement_time = time.time()
            stagnation_counter = 0
            continue
        else:
            stagnation_counter += 1
            
        # Phase 2: Swap
        if stagnation_counter > 50: # If stuck for a while, try harder swaps or restart?
            # For now, just continue swapping
            pass
            
        if len(current_set) == 0:
            break
            
        # Pick a random codeword to remove
        c_out = random.choice(current_set)
        
        # Pick a random codeword to add from the universe
        # To make it efficient, we don't want to pick from all 6561 if possible, 
        # but we need a valid cover. 
        # A good heuristic: pick a codeword that covers many currently "weakly covered" points?
        # But we are in a valid cover state (after removal phase).
        # So all points are covered.
        # We want to swap to potentially allow further reductions.
        # Let's pick a random c_in from the universe.
        c_in = random.randint(0, total - 1)
        
        if c_in == c_out:
            continue
            
        if c_in in current_set:
            continue
            
        # Tentative new set
        # We don't need to build the full set, just check if removing c_out and adding c_in maintains coverage.
        # Condition: For every y in ball[c_out], y must be covered by (current_set - {c_out}) OR y must be in ball[c_in].
        # Actually, it's simpler to just update cover_count tentatively?
        # Or check: 
        # 1. Remove c_out from cover_count.
        # 2. Check if any y in ball[c_out] becomes 0.
        # 3. If yes, check if c_in covers those y's.
        # 4. If all are covered, accept.
        
        # Let's do it explicitly for clarity and correctness, then update.
        
        # Save state of cover_count for c_out's ball? 
        # Or just check:
        # Let S' = S - {c_out} U {c_in}
        # y is covered in S' if (y in ball[c_in]) OR (y in ball[c] for some c in S-{c_out})
        
        # Efficient check:
        # 1. Temporarily decrement cover_count for ball[c_out]
        for y in balls[c_out]:
            cover_count[y] -= 1
            
        # 2. Check if any y in ball[c_out] is now 0
        uncovered_in_out_ball = [y for y in balls[c_out] if cover_count[y] == 0]
        
        if not uncovered_in_out_ball:
            # Safe to swap? 
            # Wait, if we accept, we must add c_in.
            # But if we reject, we must restore.
            # Let's check if c_in covers the uncovered ones.
            # Actually, if uncovered_in_out_ball is empty, then S-{c_out} already covers everything c_out covered.
            # So c_out was redundant? But we just tried to remove redundant ones.
            # So this case should be rare or impossible if try_remove_redundant was thorough.
            # If it happens, we can just remove c_out and keep c_out out? 
            # No, we are swapping. 
            # If c_out is redundant, we should have removed it.
            # So if uncovered_in_out_ball is empty, it means c_out is redundant.
            # We should have caught it. 
            # Let's assume it's not empty.
            
            # If it IS empty, we can just remove c_out and not add c_in? 
            # That would reduce size. But try_remove_redundant should have done that.
            # Let's just accept the swap if it maintains coverage.
            # If uncovered_in_out_ball is empty, coverage is maintained by S-{c_out}.
            # Adding c_in is optional for coverage, but we are swapping.
            # So we accept.
            pass
        else:
            # Check if c_in covers all y in uncovered_in_out_ball
            c_in_ball_set = set(balls[c_in])
            valid_swap = True
            for y in uncovered_in_out_ball:
                if y not in c_in_ball_set:
                    valid_swap = False
                    break
            
            if not valid_swap:
                # Restore cover_count
                for y in balls[c_out]:
                    cover_count[y] += 1
                continue
            else:
                # Swap is valid
                # Apply swap
                current_set.remove(c_out)
                current_set.append(c_in)
                # Update cover_count for c_in
                for y in balls[c_in]:
                    cover_count[y] += 1
                # Note: cover_count for c_out's ball was already decremented.
                # We need to increment for c_in's ball.
                # But wait, we decremented c_out's ball. 
                # If y is in both balls, it was decremented then incremented. Net 0. Correct.
                # If y is only in c_out's ball, it was decremented. It must be covered by c_in (checked above).
                # So cover_count[y] for those y's is now >= 1 (from c_in).
                # Correct.
                
                # After swap, we might have new redundancies.
                # Loop back to Phase 1.
                continue
        
        # If we get here, it means uncovered_in_out_ball was empty (c_out was redundant)
        # This implies try_remove_redundant missed it? 
        # Or maybe the state changed?
        # In any case, we have effectively removed c_out.
        # We should not add c_in if we don't have to? 
        # But the swap logic assumes we add c_in.
        # Let's just remove c_out and not add c_in to minimize size.
        # But this breaks the "swap" symmetry.
        # Let's stick to strict swap: remove c_out, add c_in.
        # If c_out was redundant, removing it reduces size. Adding c_in keeps size same.
        # So size stays same.
        # Then Phase 1 will try to remove c_in or others.
        
        # Wait, if uncovered_in_out_ball is empty, we didn't check validity for c_in.
        # But coverage is maintained by S-{c_out}. 
        # So adding c_in is safe.
        # Let's apply the swap.
        current_set.remove(c_out)
        current_set.append(c_in)
        for y in balls[c_in]:
            cover_count[y] += 1
        
        # Loop back to Phase 1

    return current_set

def _construct_ternary_8_1(q, n, radius, deadline):
    """
    Specialized constructor for q=3, n=8, radius=1.
    Uses Linear Code initialization + VNS.
    """
    total = q ** n
    balls = _precompute_balls(q, n, radius, deadline)
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    # Precompute inverse? Not strictly needed for VNS as implemented, 
    # but useful for other heuristics. We'll skip for now to save time.
    
    best_set = None
    best_size = float('inf')
    
    # Try different dimensions k for the linear code
    # k=5 -> 243 codewords. Might need expansion.
    # k=6 -> 729 codewords. Might need reduction.
    # k=7 -> 2187. Too big.
    
    start_time = time.time()
    
    # Strategy: 
    # 1. Start with k=6 linear code (729 words). 
    #    It is likely a valid cover. 
    #    Run VNS to reduce it.
    # 2. Start with k=5 linear code (243 words).
    #    It might NOT be a valid cover.
    #    If not, we need to add words to make it valid, then reduce.
    #    Or just use k=6 which is safer.
    
    # Let's try k=6 first.
    num_trials = 5
    for t in range(num_trials):
        if time.time() > deadline - 30:
            break
            
        # Generate random linear code
        initial_cw = _generate_linear_code(q, n, 6, deadline)
        
        # Check if it's a valid cover
        if not _check_coverage(initial_cw, balls, total, deadline):
            # If not, add random words until it is?
            # Or skip this trial.
            # Let's try to fix it by adding words that cover uncovered points.
            uncovered = _get_uncovered(initial_cw, balls, total)
            uncovered_indices = np.where(uncovered)[0]
            
            # Greedily add words to cover uncovered
            # For each uncovered point, pick a random word that covers it?
            # Or just pick the word itself?
            # If we pick the word itself, we add 1 word per uncovered point.
            # This might add too many.
            # Let's just skip this trial if k=6 fails (unlikely).
            continue
            
        # Run VNS
        optimized_set = _vns_search(initial_cw, balls, None, total, q, n, deadline - 10)
        
        if len(optimized_set) < best_size:
            best_size = len(optimized_set)
            best_set = set(optimized_set)
            
    # If best_set is still None, fallback to greedy
    if best_set is None:
        # Simple greedy fallback
        covered = np.zeros(total, dtype=bool)
        selected = []
        while not np.all(covered) and time.time() < deadline - 5:
            # Find uncovered word
            unc_idx = np.where(~covered)[0]
            if len(unc_idx) == 0:
                break
            # Pick a random uncovered word
            target = random.choice(unc_idx.tolist())
            # Pick a random codeword that covers it? 
            # Or just pick the target itself?
            # Picking the target itself guarantees coverage of target.
            # But it might not be optimal.
            # Let's pick a random word from the ball of the target?
            # No, let's just add the target.
            selected.append(target)
            for y in balls[target]:
                covered[y] = True
                
        best_set = set(selected)
        best_size = len(best_set)
        
    # Convert best_set to digits
    codewords = []
    for idx in best_set:
        codewords.append(_int_to_digits(idx, q, n, powers))
        
    return codewords

def construct(q, n, radius):
    start_time = time.time()
    # Evaluator kills after 150s. Use 145s deadline for safety.
    deadline = start_time + 145
    
    try:
        # Use the specialized Linear Code + VNS
        result = _construct_ternary_8_1(q, n, radius, deadline)
    except Exception as e:
        # Fallback to simple greedy if specialized fails
        try:
            # Very simple greedy
            total = q ** n
            powers = [1] * n
            for i in range(1, n):
                powers[i] = powers[i-1] * q
                
            balls = _precompute_balls(q, n, radius, deadline)
            
            covered = np.zeros(total, dtype=bool)
            selected = []
            while not np.all(covered) and time.time() < deadline - 5:
                unc_idx = np.where(~covered)[0]
                if len(unc_idx) == 0:
                    break
                target = random.choice(unc_idx.tolist())
                selected.append(target)
                for y in balls[target]:
                    covered[y] = True
                    
            codewords = []
            for idx in selected:
                codewords.append(_int_to_digits(idx, q, n, powers))
            result = codewords
        except Exception:
            # Ultimate fallback: raise
            raise e

    return result

# EVOLVE-BLOCK-END
