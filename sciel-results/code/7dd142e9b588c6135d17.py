"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n):
    if n <= 6:
        return _classic_construct(n)
    
    # For n=7, use optimized ordering of the final step
    word = _classic_construct(n - 1)
    return _optimized_step(word, n)

def _classic_construct(n):
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
    return "".join(map(str, word))

def _optimized_step(word_str, n):
    word = [int(c) for c in word_str]
    m = n
    perms, seen = [], set()
    for i in range(len(word) - m + 2):
        window = tuple(word[i:i + m - 1])
        if len(set(window)) == m - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    # Create blocks
    blocks = [list(p) + [m] + list(p) for p in perms]
    num_blocks = len(blocks)
    
    # Precompute overlap costs
    # cost[i][j] = len(block_j) - overlap(block_i, block_j)
    # We want to minimize sum of costs.
    # Overlap is max k such that suffix_k(block_i) == prefix_k(block_j)
    
    def get_overlap(i, j):
        b1 = blocks[i]
        b2 = blocks[j]
        max_k = min(len(b1), len(b2))
        for k in range(max_k, -1, -1):
            if k == 0:
                return 0
            if b1[-k:] == b2[:k]:
                return k
        return 0

    # Initialize order as 0..num_blocks-1
    order = list(range(num_blocks))
    
    # Calculate current cost
    def calc_cost(ord):
        total = len(blocks[ord[0]])
        for i in range(len(ord) - 1):
            total += len(blocks[ord[i+1]]) - get_overlap(ord[i], ord[i+1])
        return total

    current_cost = calc_cost(order)
    
    # 2-opt local search
    start_time = time.time()
    timeout = 100 # seconds
    
    while time.time() - start_time < timeout:
        improved = False
        for i in range(num_blocks - 1):
            if time.time() - start_time > timeout:
                break
            for j in range(i + 1, num_blocks):
                # Try reversing segment i..j
                # Note: 2-opt on Hamiltonian Path (not cycle)
                # We check the cost change for the edges (i-1, i) and (j, j+1)
                # and the new edges (i-1, j) and (i, j+1)
                
                # Get neighbors
                prev_i = order[i-1] if i > 0 else None
                next_j = order[j+1] if j < num_blocks - 1 else None
                
                # Old costs
                old_cost = 0
                if prev_i is not None:
                    old_cost += len(blocks[order[i]]) - get_overlap(prev_i, order[i])
                if next_j is not None:
                    old_cost += len(blocks[next_j]) - get_overlap(order[j], next_j)
                
                # New costs
                new_cost = 0
                if prev_i is not None:
                    new_cost += len(blocks[order[j]]) - get_overlap(prev_i, order[j])
                if next_j is not None:
                    new_cost += len(blocks[next_j]) - get_overlap(order[i], next_j)
                
                # Internal costs don't change for 2-opt reversal?
                # Wait, the internal edges are reversed.
                # Edge (k, k+1) becomes (k+1, k).
                # Cost is directed. Cost(u,v) != Cost(v,u) generally.
                # So we must recalculate internal costs or precompute.
                # Given N=720, internal segment length is up to 720.
                # Recalculating internal cost is O(N). Total O(N^3) per pass? Too slow.
                # We should precompute the full cost matrix.
                pass
        # If no simple 2-opt, try greedy swaps
        # Let's just do a simpler local search: swap adjacent blocks if it improves cost
        for i in range(num_blocks - 1):
            if time.time() - start_time > timeout:
                break
            # Swap order[i] and order[i+1]
            a = order[i]
            b = order[i+1]
            prev = order[i-1] if i > 0 else None
            next_ = order[i+2] if i < num_blocks - 2 else None
            
            old_cost = 0
            if prev is not None:
                old_cost += len(blocks[a]) - get_overlap(prev, a)
            old_cost += len(blocks[b]) - get_overlap(a, b)
            if next_ is not None:
                old_cost += len(blocks[next_]) - get_overlap(b, next_)
                
            new_cost = 0
            if prev is not None:
                new_cost += len(blocks[b]) - get_overlap(prev, b)
            new_cost += len(blocks[a]) - get_overlap(b, a)
            if next_ is not None:
                new_cost += len(blocks[next_]) - get_overlap(a, next_)
            
            if new_cost < old_cost:
                order[i], order[i+1] = order[i+1], order[i]
                current_cost = calc_cost(order) # Update global
                improved = True
        if not improved:
            break
            
    # Construct final word
    new_word = []
    for idx in order:
        block = blocks[idx]
        if not new_word:
            new_word = block
        else:
            overlap = get_overlap(order[order.index(idx)-1] if order.index(idx)>0 else -1, idx) # This is slow, use prev index
            # Better: iterate with previous block index
            pass
            
    # Reconstruct properly
    new_word = list(blocks[order[0]])
    for k in range(1, len(order)):
        prev_idx = order[k-1]
        curr_idx = order[k]
        b_curr = blocks[curr_idx]
        # Find overlap
        max_k = min(len(new_word), len(b_curr))
        o = 0
        for k_val in range(max_k, -1, -1):
            if k_val == 0:
                o = 0
                break
            if new_word[-k_val:] == b_curr[:k_val]:
                o = k_val
                break
        new_word += b_curr[o:]
        
    return "".join(map(str, new_word))

# EVOLVE-BLOCK-END
