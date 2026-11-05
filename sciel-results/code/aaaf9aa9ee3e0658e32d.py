import sys
import time
from itertools import permutations

_DEADLINE = 115.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def construct(n):
    global _start_time
    _start_time = time.time()
    
    if n <= 1:
        return "1"
    
    # Base case for n=2
    if n == 2:
        return "121"
    
    # We build recursively.
    # For n=3,4,5,6 we can use standard or optimized.
    # The big jump is at n=7.
    
    # Let's build the word for n-1 first.
    # Actually, the recursive structure is:
    # S_n is formed by taking the sequence of (n-1)-permutations from S_{n-1}
    # and for each perm p, adding p, then n, then p.
    # We want to order these (n-1)-permutations to maximize overlap.
    
    # Get the list of (n-1)-permutations in the order they appear in the optimal S_{n-1}.
    # For n=7, we need the order of 6-perms from S_6.
    # We can compute S_{n-1} recursively.
    
    # To save time, we can hardcode the optimal length/order for small n if needed,
    # but let's try to compute it.
    
    # For n=7, the number of 6-perms is 720.
    # We need to find a good ordering of these 720 perms.
    
    # Let's compute the "previous" word for n-1 to extract the perm order.
    # However, the standard recursive construction produces a specific order.
    # We can just generate all permutations of 1..n-1 and find a good order for them
    # independent of the previous word's internal structure, as long as we use
    # the correct set of blocks.
    # Wait, the blocks are determined by the permutations of 1..n-1.
    # The set of blocks is fixed: { p + [n] + p for p in perms(1..n-1) }.
    # The only freedom is the order of these blocks.
    
    # So for a given n, we have a set of blocks. We want to order them to minimize length.
    # This is a TSP-like problem on the overlap graph.
    
    if n == 2:
        return "121"
    
    # Get all permutations of 1..n-1
    prev_syms = list(range(1, n))
    perms = list(permutations(prev_syms))
    
    # Create blocks
    blocks = [list(p) + [n] + list(p) for p in perms]
    
    # If n is small, we can do a more exhaustive search.
    # For n=7, len(perms)=720.
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    num_perms = len(perms)
    overlap = [[0]*num_perms for _ in range(num_perms)]
    
    # Optimization: 
    # blocks[i] ends with perms[i]
    # blocks[j] starts with perms[j]
    # So overlap is the longest suffix of perms[i] that is a prefix of perms[j].
    
    for i in range(num_perms):
        if _elapsed() > _DEADLINE - 10:
            break
        pi = perms[i]
        for j in range(num_perms):
            if i == j:
                continue
            pj = perms[j]
            # Find max k
            # Max possible k is n-2
            max_k = min(len(pi), len(pj))
            # We can check from max_k down to 0
            # pi is tuple, pj is tuple
            for k in range(max_k, -1, -1):
                if k == 0 or pi[-k:] == pj[:k]:
                    overlap[i][j] = k
                    break
    
    # Greedy construction with multiple starts
    # Since 720 is small enough, we can try a few random starts or the first one.
    # Let's try to find a good path.
    
    best_order = None
    best_len = float('inf')
    
    # Try starting with each of the first 10 perms, or just the first one if time is tight.
    # 720 starts * 720 steps * 720 checks = 373M operations. Too slow in Python.
    # We need a faster heuristic.
    
    # Greedy: Start with 0. Pick next with max overlap.
    # This is O(N^2). 720^2 = 518k. Very fast.
    
    def greedy_order(start_idx):
        visited = [False] * num_perms
        order = [start_idx]
        visited[start_idx] = True
        curr = start_idx
        total_len = len(blocks[start_idx])
        
        for _ in range(num_perms - 1):
            if _elapsed() > _DEADLINE - 5:
                return None, float('inf')
            best_next = -1
            best_ov = -1
            for j in range(num_perms):
                if not visited[j]:
                    ov = overlap[curr][j]
                    if ov > best_ov:
                        best_ov = ov
                        best_next = j
            if best_next == -1:
                break
            order.append(best_next)
            visited[best_next] = True
            total_len += len(blocks[best_next]) - best_ov
            curr = best_next
        return order, total_len
    
    # Run greedy from a few starts
    starts_to_try = list(range(min(10, num_perms)))
    
    for s in starts_to_try:
        order, length = greedy_order(s)
        if order is not None and length < best_len:
            best_len = length
            best_order = order
    
    # If we have time, try a simple 2-opt improvement on the best order found.
    # 2-opt: Reverse a segment to see if it improves overlap.
    # This is complex to implement correctly for directed overlaps.
    # Let's stick to the greedy result for now.
    
    if best_order is None:
        # Fallback to standard order
        best_order = list(range(num_perms))
        
    # Construct the word
    word = []
    for idx in best_order:
        b = blocks[idx]
        # Overlap with current end
        max_ov = min(len(word), len(b))
        ov = 0
        for o in range(max_ov, -1, -1):
            if o == 0 or word[-o:] == b[:o]:
                ov = o
                break
        word.extend(b[ov:])
        
    return "".join(map(str, word))
