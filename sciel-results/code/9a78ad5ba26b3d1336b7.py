import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # The problem: Find longest subset S of {2, ..., 128} such that 
    # all products of consecutive blocks in sorted(S) are distinct.
    
    # Helper: Check validity of a sequence
    # Returns (is_valid, list_of_colliding_pairs_of_indices)
    # A collision means product(seq[i:j+1]) == product(seq[k:l+1]) for i!=k or j!=l
    
    def get_collisions(seq):
        """
        Returns a list of tuples (i, j, k, l) representing collisions.
        Also returns a set of 'bad' indices involved in collisions.
        """
        n = len(seq)
        if n == 0:
            return [], set()
        
        # Calculate all block products
        # prod[i][j] = product of seq[i..j]
        # To save memory/time, we can just compute on the fly or use a dict
        # Since n is small (<= 127), O(n^2) products is fine.
        prods = {}
        collisions = []
        bad_indices = set()
        
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    # Collision found
                    # prods[p] stores the (start, end) of the previous block with this product
                    prev_i, prev_j = prods[p]
                    # Record collision
                    collisions.append((prev_i, prev_j, i, j))
                    # Mark indices involved as bad
                    for idx in range(prev_i, prev_j + 1):
                        bad_indices.add(idx)
                    for idx in range(i, j + 1):
                        bad_indices.add(idx)
                else:
                    prods[p] = (i, j)
                    
        return collisions, bad_indices

    def is_valid(seq):
        _, bad = get_collisions(seq)
        return len(bad) == 0

    # Strategy: 
    # 1. Generate a good initial sequence using a greedy approach that tries to maximize density.
    # 2. Use Local Search: 
    #    - Identify elements involved in collisions.
    #    - Try removing the "most conflicting" element.
    #    - Try inserting an element that was previously removed, if it doesn't cause new collisions 
    #      or resolves more than it creates.
    #    - This is essentially Simulated Annealing or Hill Climbing on the subset space.

    # Initial Greedy Construction
    # We want a dense start. A simple greedy picking smallest available is a good baseline.
    def greedy_init():
        seq = []
        # We'll try to build a sequence. 
        # To make it denser, we can try multiple random orderings or just standard ascending.
        # Standard ascending greedy:
        current_seq = []
        prods = set()
        suffix_prods = [1] # dummy, we manage carefully
        
        # Actually, let's use the same efficient check as before for init
        # But let's try a slightly different greedy: 
        # Iterate 2..128, if adding x keeps it valid, add it.
        # This usually yields ~40-50 elements.
        
        # Let's try to get a higher density start by randomizing the order slightly?
        # No, strictly increasing is required. So the order is fixed by values.
        # The only choice is which values to include.
        
        # Greedy: Include if valid.
        for x in range(2, 129):
            # Check if adding x to current_seq is valid
            # New products: x, x*last, x*last*second_last...
            new_prods = [x]
            valid = True
            if x in prods:
                valid = False
            else:
                prod = x
                for i in range(len(current_seq) - 1, -1, -1):
                    prod *= current_seq[i]
                    if prod in prods:
                        valid = False
                        break
                    new_prods.append(prod)
            
            if valid:
                current_seq.append(x)
                prods.update(new_prods)
                
        return current_seq

    best_seq = greedy_init()
    best_len = len(best_seq)
    
    # Local Search Loop
    # We will maintain a set of "removed" candidates to potentially re-insert.
    # We define a move as:
    # 1. Remove an element from current_seq.
    # 2. Insert an element from removed_set into current_seq (maintaining order).
    
    # To make this efficient, we need fast validity checks.
    # Given the small size (N=127), O(N^2) check per move is acceptable if we limit moves.
    # 127^2 * 1000 moves * 100 restarts = 1.6 * 10^9 ops... might be too slow in Python.
    # Let's optimize the check or limit the search.
    
    # Better approach for Local Search:
    # Keep the set of products.
    # When removing an element, we don't easily "subtract" products.
    # So, we will re-calculate validity from scratch for the candidate sequence after each move.
    # Given N~50, N^2 = 2500 products. 
    # If we do 1000 moves per second, that's 2.5M ops/sec. 55s -> 137M ops. Feasible.
    
    def local_search(start_seq, time_limit):
        current_seq = start_seq[:]
        current_len = len(current_seq)
        removed = [x for x in range(2, 129) if x not in current_seq]
        
        # Helper to check validity and return bad indices
        def check_validity(seq):
            n = len(seq)
            if n == 0: return True, set()
            prods = {}
            bad = set()
            for i in range(n):
                p = 1
                for j in range(i, n):
                    p *= seq[j]
                    if p in prods:
                        pi, pj = prods[p]
                        for k in range(pi, pj+1): bad.add(k)
                        for k in range(i, j+1): bad.add(k)
                    else:
                        prods[p] = (i, j)
            return len(bad)==0, bad

        # We track the best sequence found in this LS run
        best_local_seq = current_seq[:]
        best_local_len = current_len
        
        while time.time() < time_limit:
            # Identify conflicts
            is_val, bad_indices = check_validity(current_seq)
            
            if is_val:
                # If valid, try to insert something from 'removed'
                # Pick a random element from removed, or the one that causes fewest new conflicts
                if not removed:
                    break
                
                # Heuristic: Try inserting the smallest removed element? 
                # Or random? Let's try a few random candidates.
                improved = False
                for _ in range(10):
                    if not removed: break
                    idx_to_insert = random.randrange(len(removed))
                    val = removed[idx_to_insert]
                    
                    # Find position in current_seq
                    # Since current_seq is sorted, we can bisect
                    import bisect
                    pos = bisect.bisect_left(current_seq, val)
                    
                    # Create candidate
                    candidate = current_seq[:pos] + [val] + current_seq[pos:]
                    
                    # Check validity
                    cand_val, _ = check_validity(candidate)
                    if cand_val:
                        current_seq = candidate
                        current_len += 1
                        removed.pop(idx_to_insert)
                        improved = True
                        
                        if current_len > best_local_len:
                            best_local_len = current_len
                            best_local_seq = current_seq[:]
                        break
                    
                if not improved:
                    # If we can't insert, maybe we should remove something to allow better insertions?
                    # Or just break if we're stuck?
                    # Let's try removing a random element to "shake" the state if we've been stuck.
                    # Actually, if we are valid, we are in a local optimum for insertion.
                    # We might need to remove an element to make room for a "better" set.
                    # This is hard to judge. 
                    # For now, if valid and can't insert, we accept this as a local optimum.
                    break

            else:
                # Not valid. We must remove at least one element from bad_indices.
                # Heuristic: Remove the element that is in the most collisions?
                # Or just pick one from bad_indices.
                # Let's pick the one that appears in the most blocks involved in collisions.
                # For simplicity, pick a random one from bad_indices.
                # Or, pick the one with the highest "degree" in the conflict graph.
                
                # Let's count how many bad indices each element is part of.
                # Actually, bad_indices are indices in the sequence.
                # We want to remove a value.
                # Let's pick the value at a random bad index.
                # To be smarter: pick the value that is in the most collisions.
                
                # Re-calculate collisions to get frequency?
                # We already have bad_indices.
                # Let's just pick a random one from bad_indices to remove.
                # This is a random walk on the space of valid/invalid subsets.
                
                idx_to_remove = random.choice(list(bad_indices))
                val_to_remove = current_seq[idx_to_remove]
                
                # Remove it
                current_seq.pop(idx_to_remove)
                current_len -= 1
                removed.append(val_to_remove)
                removed.sort() # Keep removed sorted for bisect? Not strictly necessary for random pick, but good.
                
                # If we removed something, we might have created new space.
                # The loop will continue and try to insert or remove again.
                
                # Update best if we are valid? (We are not valid after removal necessarily, 
                # but removing a conflicting element *should* help, though it might not resolve all conflicts).
                # Actually, removing one element from a collision pair might leave the other product unique, 
                # or might cause a new collision if that product was unique before? No, removing an element 
                # only removes products, it doesn't create new ones. So if we had a collision A==B, 
                # and we remove an element from A, A is gone. B remains. No new collision is created 
                # by removal. 
                # Wait, is that true?
                # Products are of consecutive blocks. 
                # If we remove x, the sequence splits. Blocks that spanned x are gone. 
                # New blocks are formed? No, the sequence is just shorter. 
                # The set of products is a subset of the previous set of products?
                # No. If we have [a, b, c] and remove b, we get [a, c]. 
                # Products before: a, b, c, ab, bc, abc.
                # Products after: a, c, ac.
                # ac is a NEW product! It wasn't in the previous set.
                # So removal CAN create new collisions.
                
                # So we must re-check validity.
                
                # If current_len > best_local_len:
                #    ... (can't happen, we removed)
                
        return best_local_seq, best_local_len

    # Run multiple local search restarts
    # We can vary the initial sequence slightly or just use the greedy one.
    # To diversify, we can try greedy with different "noise" or just run LS multiple times 
    # on the same start, but LS is deterministic-ish if random seed is fixed. 
    # So we need randomization in LS.
    
    random.seed(42)
    
    global_best = best_seq
    global_best_len = best_len
    
    # Run LS for a fixed number of iterations or until time runs out
    # Each LS run might take some time. 
    # Let's run LS in chunks.
    
    while time.time() < deadline - 5.0:
        # Start a new LS run from the current global best
        # Or from a perturbed version?
        # Let's start from global best and perturb it slightly (remove 5 random elements) to escape local optima.
        
        perturbed = global_best[:]
        # Remove 5 random elements
        for _ in range(min(5, len(perturbed))):
            if not perturbed: break
            idx = random.randrange(len(perturbed))
            perturbed.pop(idx)
            
        # Run local search on perturbed
        # Limit time per LS run to avoid getting stuck
        ls_deadline = time.time() + 5.0 
        if ls_deadline > deadline - 2.0:
            ls_deadline = deadline - 2.0
            
        seq, length = local_search(perturbed, ls_deadline)
        
        if length > global_best_len:
            global_best_len = length
            global_best = seq
    
    # Final check: try to fill gaps in global_best with a simple greedy pass?
    # If global_best is valid, we can try to insert any remaining numbers that fit.
    if is_valid(global_best):
        # Try to insert any missing numbers
        missing = [x for x in range(2, 129) if x not in global_best]
        # Sort missing
        # Try to insert each one if it keeps validity
        # This is a simple improvement step
        for x in missing:
            if time.time() > deadline - 1.0: break
            import bisect
            pos = bisect.bisect_left(global_best, x)
            candidate = global_best[:pos] + [x] + global_best[pos:]
            if is_valid(candidate):
                global_best = candidate
                global_best_len += 1

    return global_best

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
