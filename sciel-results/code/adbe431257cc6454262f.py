"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # For n <= 6, the classic construction is optimal or close enough and fast.
    # For n = 7, we attempt local search to improve the order of permutations.
    
    if n <= 6:
        return _classic_superperm(n)
    
    # --- Step 1: Build SP(n-1) and extract permutation order ---
    # We need the list of permutations of 1..n-1 in the order they appear in the classic SP(n-1).
    # Let k = n-1.
    k = n - 1
    word_k = _classic_superperm_list(k)
    
    # Extract unique permutations of length k from word_k
    perms_k = []
    seen = set()
    for i in range(len(word_k) - k + 1):
        window = tuple(word_k[i:i + k])
        # Check if it's a valid permutation of 1..k
        # Since word_k is a superpermutation, all windows of size k that are permutations are unique?
        # Actually, in SP(k), each permutation appears exactly once as a window of size k?
        # No, SP(k) contains every permutation of 1..k as a substring of length k.
        # Are they unique? Yes, by definition of superpermutation, every permutation appears.
        # Can a permutation appear twice? Yes, but we want the order of *first* appearance?
        # Or the order of all appearances?
        # The standard construction SP(n) is built by iterating through the permutations of SP(n-1)
        # in the order they appear in SP(n-1).
        # Specifically, if SP(n-1) = w, and the permutations of 1..n-1 appear in w at indices i_1, i_2, ..., i_{(n-1)!},
        # then SP(n) is formed by concatenating blocks P_i + [n] + P_i.
        # The "order" is the sequence of these distinct permutations as they are encountered in the scan of w.
        
        if window not in seen:
            # Verify it's a permutation of 1..k
            # For k=6, elements are 1..6.
            if sorted(window) == list(range(1, k+1)):
                seen.add(window)
                perms_k.append(window)
    
    num_perms = len(perms_k)
    # For n=7, k=6, num_perms should be 720.
    if num_perms != k * (k-1) * (k-2) * (k-3) * (k-4) * (k-5):
        # Fallback if extraction failed
        return _classic_superperm(n)

    # --- Step 2: Precompute Overlap Matrix ---
    # M[i][j] = length of longest suffix of perms_k[i] matching prefix of perms_k[j]
    # This allows us to calculate the length of the resulting word for a given order.
    # The total length of the word constructed from an order [p_1, p_2, ..., p_m] is:
    # L = sum(len(block_i)) - sum(overlap(i, i+1))
    # block_i = p_i + [n] + p_i, so len(block_i) = 2k + 1.
    # Total L = m * (2k + 1) - sum(overlap(i, i+1))
    # To minimize L, we maximize sum(overlap(i, i+1)).
    
    # Overlap is between the end of the *current word* and the start of the *next block*.
    # However, the current word ends with ... p_i.
    # The next block starts with p_{i+1}.
    # So the overlap is indeed the longest suffix of p_i matching prefix of p_{i+1}.
    # Wait, is it possible for the overlap to extend further back?
    # The block is p_i + [n] + p_i.
    # The previous block ended with ... p_{i-1} + [n] + p_{i-1}.
    # The suffix of the current word is p_i.
    # The prefix of the next block is p_{i+1}.
    # So we are matching suffix of p_i with prefix of p_{i+1}.
    # Since p_i and p_{i+1} are permutations of 1..k, they have no repeated elements within themselves.
    # The maximum overlap is k (if p_i == p_{i+1}, but they are distinct, so max is k-1? No, if they are distinct, they can't match fully. Max is k-1? 
    # Actually, if p_i ends with a sequence that matches the start of p_{i+1}.
    # Example: p_i = (1,2,3), p_{i+1} = (2,3,4). Overlap is 2 (2,3).
    # So M[i][j] is well-defined and in [0, k-1].
    
    M = [[0] * num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        p1 = perms_k[i]
        for j in range(num_perms):
            p2 = perms_k[j]
            # Max possible overlap is k-1 (since they are distinct perms, they can't be identical, so full match k is impossible? 
            # Wait, if p1 = (1,2,3) and p2 = (1,2,3), overlap is 3. But i != j implies p1 != p2.
            # So max overlap is k-1?
            # If p1 ends with (a,b) and p2 starts with (a,b), overlap is 2.
            # Can overlap be k? Only if p1 == p2. Since we extracted unique perms, p1 != p2 for i != j.
            # So max overlap is k-1.
            for o in range(k - 1, -1, -1):
                if o == 0:
                    M[i][j] = 0
                    break
                # Compare last o elements of p1 with first o elements of p2
                if p1[k - o:] == p2[:o]:
                    M[i][j] = o
                    break
    
    # --- Step 3: Local Search to Maximize Sum of Overlaps ---
    # Initial order: the natural order from SP(k)
    order = list(range(num_perms))
    
    def calc_score(ord):
        score = 0
        for i in range(len(ord) - 1):
            score += M[ord[i]][ord[i+1]]
        return score
    
    best_order = order[:]
    best_score = calc_score(order)
    
    random.seed(42)
    
    # Hill Climbing / Simulated Annealing
    # We have ~110s.
    # calc_score is O(N) where N=720. Very fast.
    # We can do many iterations.
    
    current_order = order[:]
    current_score = best_score
    
    # Use a simple restart strategy if stuck
    # Or just aggressive hill climbing with random moves.
    
    # Move types:
    # 1. Swap adjacent
    # 2. Move element i to position j
    
    # To speed up delta calculation, we can do it incrementally, but for N=720, full recalc is ~720 ops.
    # 110s / (720 ops * 1e-9 s/op) -> millions of iterations possible.
    # Let's just do full recalc for simplicity and robustness, unless it's too slow.
    # Python overhead is higher. Let's assume ~1000 iterations/sec for full recalc?
    # 110 * 1000 = 110,000 iterations. Should be enough to find improvements.
    
    while time.time() < deadline:
        # Choose move type
        if random.random() < 0.5:
            # Swap adjacent
            i = random.randint(0, num_perms - 2)
            new_order = current_order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            new_score = calc_score(new_order)
            
            if new_score > best_score:
                best_score = new_score
                best_order = new_order
            if new_score >= current_score:
                current_order = new_order
                current_score = new_score
        else:
            # Move element i to position j
            i = random.randint(0, num_perms - 1)
            j = random.randint(0, num_perms - 1)
            if i == j:
                continue
            new_order = current_order[:]
            item = new_order.pop(i)
            new_order.insert(j, item)
            new_score = calc_score(new_order)
            
            if new_score > best_score:
                best_score = new_score
                best_order = new_order
            if new_score >= current_score:
                current_order = new_order
                current_score = new_score
        
        # Optional: Restart if stuck for too long?
        # Keep it simple for now.
    
    # --- Step 4: Construct Final Word from Best Order ---
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perms_k[p_idx]
        block = list(p) + [n] + list(p)
        
        if idx == 0:
            word = block
        else:
            # Find overlap with current word
            # Suffix of word vs Prefix of block
            # We know the overlap is M[best_order[idx-1]][best_order[idx]]
            # But to be safe and correct, we should verify or just use the calculated value?
            # The calculated value M[i][j] is the overlap between p_i and p_j.
            # The word ends with p_i. The block starts with p_j.
            # So the overlap is exactly M[i][j].
            # However, could there be a larger overlap involving the [n] symbol?
            # No, because p_i consists of 1..k and block starts with 1..k.
            # The [n] is in the middle of the block.
            # The suffix of the word is ... p_i.
            # The prefix of the block is p_j ...
            # So we are matching suffix of p_i with prefix of p_j.
            # This is exactly what M[i][j] computes.
            
            ov = M[best_order[idx-1]][best_order[idx]]
            # Append block[ov:]
            word += block[ov:]
            
    return "".join(map(str, word))

def _classic_superperm_list(n):
    """Returns the classic superpermutation for n as a list of integers."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word

def _classic_superperm(n):
    """Returns the classic superpermutation for n as a string."""
    return "".join(map(str, _classic_superperm_list(n)))

# EVOLVE-BLOCK-END
