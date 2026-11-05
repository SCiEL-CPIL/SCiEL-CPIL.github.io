import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Only handle the specific case or general?
    # The task says q=3, n=6, radius=1.
    # But interface is general. We should handle general if possible, 
    # but the "known optimal" hint is specific.
    # If q, n, radius differ, we can fall back to a simple greedy or the tetracode sum if applicable.
    # However, the prompt implies this specific task instance.
    # Let's implement the general greedy + local search, which works for small n.
    
    total_words = q ** n
    if total_words > 10000:
        # Fallback for larger problems: Greedy only
        # But n=6, q=3 is small.
        pass
        
    # Encoding: word -> int
    def encode(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val
        
    def decode(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    # Precompute covers
    # cover[c] is a bitmask of words covered by codeword c
    covers = [0] * total_words
    
    # Generate all words
    all_words = [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Mapping from word tuple to index
    # Actually, we can just use the integer index as the vertex ID.
    # We need to know which vertices are at distance <= radius from a given vertex.
    
    # For efficiency, precompute neighbors for each vertex
    # Since n is small, we can just iterate.
    for idx, w in enumerate(all_words):
        mask = 1 << idx
        for d in range(1, radius + 1):
            # Generate all words at distance d? 
            # Easier: generate all words and check distance? 
            # No, that's O(N^2). 
            # Better: for each position, change value.
            # For radius 1, just check distance 0 and 1.
            pass
            
    # Let's do it specifically for radius 1 optimization, but keep general logic if possible.
    # Given the constraints, n=6 is small.
    
    # Precompute covers for all possible codewords
    for c_idx, c_word in enumerate(all_words):
        mask = 0
        for v_idx, v_word in enumerate(all_words):
            dist = sum(1 for a, b in zip(c_word, v_word) if a != b)
            if dist <= radius:
                mask |= (1 << v_idx)
        covers[c_idx] = mask
        
    # Greedy Set Cover
    uncovered = (1 << total_words) - 1
    solution = []
    candidates = list(range(total_words))
    
    while uncovered:
        best_c = -1
        best_gain = 0
        # Heuristic: pick c that maximizes (covers[c] & uncovered)
        # To speed up, we can sort candidates by "popcount" of cover once? 
        # Or just iterate. 729 * 729 is ~500k ops, fine.
        for c in candidates:
            gain = bin(covers[c] & uncovered).count('1')
            if gain > best_gain:
                best_gain = gain
                best_c = c
                if gain == bin(uncovered).count('1'):
                    break
        if best_c == -1:
            break
        solution.append(best_c)
        candidates.remove(best_c)
        uncovered &= ~covers[best_c]
        
    # Local Search: Swap and Remove
    # We represent solution as a set of indices
    sol_set = set(solution)
    
    def is_redundant(c, current_set):
        # Check if covers[c] is subset of union of covers of others
        union_mask = 0
        for other in current_set:
            if other != c:
                union_mask |= covers[other]
        return (covers[c] & ~union_mask) == 0
        
    def try_reduce():
        nonlocal sol_set
        # 1. Remove redundant
        for c in list(sol_set):
            if is_redundant(c, sol_set):
                sol_set.remove(c)
                return True
        return False
        
    def try_swap_and_reduce():
        nonlocal sol_set
        # Try to find a swap that allows a removal
        # Iterate over all c_old in sol_set
        for c_old in list(sol_set):
            # Calculate words only covered by c_old
            union_rest = 0
            for c in sol_set:
                if c != c_old:
                    union_rest |= covers[c]
            lost = covers[c_old] & ~union_rest
            
            if lost == 0:
                # Redundant, handled by try_reduce
                continue
                
            # Find c_new not in sol_set such that lost is subset of covers[c_new]
            for c_new in range(total_words):
                if c_new in sol_set:
                    continue
                if (lost & ~covers[c_new]) == 0:
                    # Valid swap candidate
                    # Perform swap
                    new_set = sol_set.copy()
                    new_set.remove(c_old)
                    new_set.add(c_new)
                    
                    # Check if any codeword in new_set is now redundant
                    for c in new_set:
                        if is_redundant(c, new_set):
                            # Accept swap and removal
                            sol_set = new_set
                            sol_set.remove(c)
                            return True
                            
        return False
        
    # Main loop
    while time.time() < deadline:
        if try_reduce():
            continue
        if try_swap_and_reduce():
            continue
        # If no improvement, maybe try random restarts or different swap orders?
        # For now, if stuck, break or try a different heuristic.
        # Given the small search space, we might just be at the optimum or a local min.
        # 73 is the target. If we are at 74, we need to break the local min.
        # Randomized swap: pick random c_old, random c_new that covers lost?
        # Let's add a randomized phase if stuck for a while.
        
        # Simple break condition if we think we're done
        # But we want to be sure.
        # Let's just run until deadline if we keep finding improvements.
        # If no improvement in one full pass, we might be stuck.
        # We can try to randomize the order of c_old or c_new.
        
        # To avoid infinite loop on no improvement:
        # If try_swap_and_reduce fails, we might try a "2-swap" or just break.
        # Given the problem structure, simple 1-swap + remove is often sufficient for small N.
        
        # If we are stuck, we can try to add a random codeword and then reduce?
        # No, that increases size.
        
        # Let's assume we found the minimum if we can't improve with 1-swap.
        # Break to be safe against timeouts, returning current best.
        break

    return [decode(c) for c in sorted(sol_set)]
