import random
import time

def construct(q, n, radius):
    # For the specific task q=3, n=8, radius=1
    # General implementation for small q, n
    
    total_words = q ** n
    
    # Precompute neighbors for each word index
    # index -> list of indices within Hamming distance <= radius
    # To do this efficiently, we can iterate through all words and generate neighbors
    # Or, for each word, generate all words at distance <= radius.
    
    # Helper to convert index to tuple
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    # Helper to convert tuple to index
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbors
    # This might be memory heavy for large n, but for n=8, q=3, it's fine.
    # 3^8 = 6561 words.
    # Neighbors for R=1: 1 + 8*2 = 17 per word.
    # Total edges ~ 111k.
    
    neighbors = [[] for _ in range(total_words)]
    
    # Generate all words
    all_words = [idx_to_tuple(i) for i in range(total_words)]
    
    # For each word, find its neighbors
    # To speed up, we can just generate neighbors on the fly during the greedy step?
    # No, precomputation is better for local search.
    
    # Efficient neighbor generation:
    # For each word w, for each position i, for each value v != w[i]:
    #   create new word, get index, add to neighbors[w_idx]
    
    for i, w in enumerate(all_words):
        neighbors[i].append(i) # Self
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Create new word
                # Instead of creating tuple and converting, we can compute index difference
                # diff = (new_val - orig_val) * (q ** (n - 1 - pos))
                # new_idx = i + diff
                diff = (new_val - orig_val) * (q ** (n - 1 - pos))
                neighbors[i].append(i + diff)
                
    # Greedy Initialization
    covered = [False] * total_words
    centers = []
    
    # Count uncovered
    uncovered_count = total_words
    
    # To speed up greedy, we can maintain a list of uncovered words?
    # Or just iterate. 6561 is small.
    
    while uncovered_count > 0:
        best_idx = -1
        best_gain = -1
        
        # We need to find the center that covers the most *currently uncovered* words.
        # We can iterate through all words 0..total_words-1
        for i in range(total_words):
            # Calculate gain
            gain = 0
            for nb in neighbors[i]:
                if not covered[nb]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_gain == 0:
            # Should not happen if uncovered_count > 0
            break
            
        centers.append(best_idx)
        
        # Update covered
        for nb in neighbors[best_idx]:
            if not covered[nb]:
                covered[nb] = True
                uncovered_count -= 1
                
    # Local Search to minimize centers
    # Maintain cover_count
    cover_count = [0] * total_words
    for c in centers:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    # Helper to check if a center is redundant
    def is_redundant(c_idx):
        for nb in neighbors[c_idx]:
            if cover_count[nb] == 1:
                return False
        return True

    # Helper to remove a center
    def remove_center(c_idx):
        for nb in neighbors[c_idx]:
            cover_count[nb] -= 1
            
    # Helper to add a center
    def add_center(c_idx):
        for nb in neighbors[c_idx]:
            cover_count[nb] += 1

    # Optimization Loop
    start_time = time.time()
    max_time = 115.0 # Leave 5s margin
    
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        
        # 1. Reduction Phase
        # Try to remove any redundant center
        # Randomize order to avoid bias
        random.shuffle(centers)
        for i in range(len(centers)):
            if time.time() - start_time > max_time:
                break
            c_idx = centers[i]
            if is_redundant(c_idx):
                remove_center(c_idx)
                centers.pop(i)
                improved = True
                # Restart reduction as removal of one might make others redundant
                # Or just continue? Continuing is fine, but restarting is safer for correctness of "improved" flag logic
                # Actually, if we pop, indices shift.
                # Let's restart the inner loop to be safe
                break
        
        if time.time() - start_time > max_time:
            break

        if improved:
            continue
            
        # 2. Swap Phase
        # If no center is redundant, try to swap one to create a redundancy.
        # Pick a random center c.
        # Try to swap c with w.
        # Condition for valid swap:
        # After removing c and adding w, all words must still be covered.
        # This means:
        # For all nb in neighbors[c]:
        #   if cover_count[nb] == 1 (only covered by c):
        #       then w must cover nb (i.e., nb in neighbors[w])
        
        # To make this efficient, we can precompute for each c, the set of "critical" neighbors
        # critical_nb = [nb for nb in neighbors[c] if cover_count[nb] == 1]
        # If critical_nb is empty, c is redundant (handled above).
        # So critical_nb is not empty.
        # We need w such that critical_nb is a subset of neighbors[w].
        
        # This is a set intersection problem.
        # Given small size, we can just iterate w?
        # 6561 * 6561 is 43M. Too slow for many iterations.
        
        # Heuristic:
        # Pick a random center c.
        # Get its critical neighbors.
        # If len(critical_nb) is small, find w covering them.
        # If len(critical_nb) is large, hard to find w.
        
        # Let's try a few random swaps per iteration.
        for _ in range(10): # Try 10 random swaps
            if time.time() - start_time > max_time:
                break
            if not centers:
                break
            c_idx = random.choice(centers)
            critical_nb = []
            for nb in neighbors[c_idx]:
                if cover_count[nb] == 1:
                    critical_nb.append(nb)
                    
            if not critical_nb:
                # Should have been removed in reduction phase
                remove_center(c_idx)
                centers.remove(c_idx)
                improved = True
                break
                
            # We need a w that covers all critical_nb.
            # w must be in the intersection of neighbors of all critical_nb.
            # Let's pick the first critical neighbor nb0.
            # w must be in neighbors[nb0].
            # Check if w covers other critical neighbors.
            
            # To speed up, we can just pick a random w from neighbors[critical_nb[0]]
            # and check if it covers all critical_nb.
            
            candidate_pool = neighbors[critical_nb[0]]
            if not candidate_pool:
                continue
                
            w_idx = random.choice(candidate_pool)
            
            # Check if w_idx covers all critical_nb
            valid_swap = True
            # Optimization: Only check critical_nb, not all neighbors
            # But we also need to ensure w doesn't "break" anything?
            # Adding w only increases counts. Removing c decreases counts.
            # The only risk is words that were ONLY covered by c.
            # Those are exactly critical_nb.
            # So if w covers all critical_nb, the swap is valid.
            
            for nb in critical_nb:
                # Check if nb is in neighbors[w_idx]
                # Since neighbors are precomputed, we can use a set for fast lookup?
                # Or just iterate?
                # neighbors[w_idx] is small (17 for R=1).
                if nb not in neighbors[w_idx]:
                    valid_swap = False
                    break
                    
            if valid_swap:
                # Perform swap
                remove_center(c_idx)
                add_center(w_idx)
                centers.remove(c_idx)
                if w_idx not in centers:
                    centers.append(w_idx)
                else:
                    # w_idx was already a center?
                    # If w_idx is already in centers, then we are just removing c_idx.
                    # This is a reduction, not a swap.
                    # But we checked is_redundant(c_idx) and it was false.
                    # So w_idx cannot be in centers if it covers critical_nb?
                    # If w_idx is in centers, then critical_nb is covered by w_idx AND c_idx.
                    # So cover_count[critical_nb] >= 2.
                    # Contradiction with cover_count == 1.
                    # So w_idx is NOT in centers.
                    pass
                improved = True
                break
                
    # Convert indices back to lists of integers
    result = [all_words[c] for c in centers]
    # The problem asks for list of codewords, each a list of n integers
    # all_words contains tuples. Convert to lists.
    return [list(w) for w in result]
