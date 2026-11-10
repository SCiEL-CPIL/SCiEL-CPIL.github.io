"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from functools import lru_cache

# EVOLVE-BLOCK-START
def _build_superperm(n):
    """Build a superpermutation using recursive construction with overlap optimization."""
    word = [1]
    for m in range(2, n + 1):
        # Extract all distinct (m-1)-permutations in order of appearance
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build new word by inserting m into each permutation
        # For each perm P, the block is P + [m] + P
        # We want to maximize overlap between consecutive blocks
        new = []
        for idx, p in enumerate(perms):
            block = list(p) + [m] + list(p)
            if not new:
                new = block
            else:
                # Find maximum overlap: suffix of new matches prefix of block
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, 0, -1):
                    if new[-o:] == block[:o]:
                        overlap = o
                        break
                new.extend(block[overlap:])
        word = new
    return word

def _overlap_cost(a, b):
    """Compute the number of new symbols needed to append b after a."""
    # Find maximum suffix of a that matches prefix of b
    max_ov = min(len(a), len(b))
    for o in range(max_ov, 0, -1):
        if a[-o:] == b[:o]:
            return len(b) - o
    return len(b)

def _build_from_order(perms, m):
    """Build the superpermutation word given an ordering of (m-1)-permutations."""
    word = []
    for p in perms:
        block = list(p) + [m] + list(p)
        if not word:
            word = block
        else:
            max_ov = min(len(word), len(block))
            ov = 0
            for o in range(max_ov, 0, -1):
                if word[-o:] == block[:o]:
                    ov = o
                    break
            word.extend(block[ov:])
    return word

def _lin_kernighan_optimize(order, cost_matrix, max_iters=50):
    """
    Apply Lin-Kernighan style 2-opt/3-opt moves to improve the ordering.
    order: list of indices into perms
    cost_matrix: cost_matrix[i][j] = cost of transitioning from perm i to perm j
    Returns improved order.
    """
    n = len(order)
    if n <= 2:
        return order[:]
    
    current_order = order[:]
    current_cost = sum(cost_matrix[current_order[i]][current_order[i+1]] for i in range(n-1))
    
    for iteration in range(max_iters):
        improved = False
        # Try 2-opt moves: reverse a segment
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Cost change for reversing segment i..j
                # Old edges: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
                # New edges: (i-1, j), (j, j-1), ..., (i+1, i), (i, i+1)
                # Wait, 2-opt reverses the segment, so:
                # Old: ...A, B, C, D... where A=order[i-1], B=order[i], C=order[j], D=order[j+1]
                # New: ...A, C, B, D... (reverse B..C)
                
                old_cost = 0
                new_cost = 0
                
                # Edge before segment
                if i > 0:
                    old_cost += cost_matrix[current_order[i-1]][current_order[i]]
                    new_cost += cost_matrix[current_order[i-1]][current_order[j]]
                
                # Edges within segment
                for k in range(i, j):
                    old_cost += cost_matrix[current_order[k]][current_order[k+1]]
                    new_cost += cost_matrix[current_order[k+1]][current_order[k]]
                
                # Edge after segment
                if j < n - 1:
                    old_cost += cost_matrix[current_order[j]][current_order[j+1]]
                    new_cost += cost_matrix[current_order[i]][current_order[j+1]]
                
                if new_cost < old_cost:
                    # Apply the reversal
                    current_order[i:j+1] = current_order[i:j+1][::-1]
                    current_cost += new_cost - old_cost
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return current_order

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Build the standard recursive superpermutation
    word = _build_superperm(n)
    
    if n >= 3:
        # Build up to n-1
        word_n1 = _build_superperm(n - 1)
        
        # Now insert n into word_n1 with optimized overlaps
        m = n
        perms = []
        seen = set()
        for i in range(len(word_n1) - (m - 1) + 1):
            window = tuple(word_n1[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Standard construction as baseline
        standard_word = _build_from_order(perms, m)
        best_word = standard_word
        best_len = len(best_word)
        
        # Build cost matrix for Lin-Kernighan optimization
        # cost_matrix[i][j] = number of new symbols to append perm[j] after perm[i]
        num_perms = len(perms)
        
        # For n=7, num_perms = 720, so cost matrix is 720x720 = 518400 entries
        # This is manageable
        cost_matrix = [[0] * num_perms for _ in range(num_perms)]
        
        for i in range(num_perms):
            block_i = list(perms[i]) + [m] + list(perms[i])
            for j in range(num_perms):
                if i == j:
                    continue
                block_j = list(perms[j]) + [m] + list(perms[j])
                cost_matrix[i][j] = _overlap_cost(block_i, block_j)
        
        # Standard order is just 0, 1, 2, ..., num_perms-1
        standard_order = list(range(num_perms))
        
        # Apply Lin-Kernighan optimization
        if time.time() < deadline - 30:
            optimized_order = _lin_kernighan_optimize(standard_order, cost_matrix, max_iters=100)
            optimized_word = _build_from_order([perms[i] for i in optimized_order], m)
            if len(optimized_word) < best_len:
                best_word = optimized_word
                best_len = len(best_word)
        
        # Try a few random restarts with local optimization
        if time.time() < deadline - 20:
            random.seed(42)
            for restart in range(5):
                if time.time() >= deadline - 15:
                    break
                
                # Create a perturbed order
                perturbed = standard_order[:]
                # Do a few random 2-opt moves
                for _ in range(10):
                    i = random.randint(0, num_perms - 2)
                    j = random.randint(i + 1, num_perms - 1)
                    perturbed[i:j+1] = perturbed[i:j+1][::-1]
                
                # Optimize from this starting point
                optimized = _lin_kernighan_optimize(perturbed, cost_matrix, max_iters=50)
                word_candidate = _build_from_order([perms[i] for i in optimized], m)
                if len(word_candidate) < best_len:
                    best_word = word_candidate
                    best_len = len(word_candidate)
        
        word = best_word
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
