import time
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    if n <= 1:
        return "1"
    
    # 1. Generate standard superpermutation for n-1
    # We need the list of blocks (permutations of 1..n-1) in the order they appear
    # and the full word to verify or just to get the blocks.
    # Actually, we can just generate the blocks recursively.
    
    def get_blocks(n_val):
        if n_val == 1:
            return [[1]]
        prev_blocks = get_blocks(n_val - 1)
        blocks = []
        # The standard construction order is determined by the appearance in the word.
        # We can simulate the word construction to get the order, or just use the known property.
        # The order of blocks for n is the same as the order of permutations in the n-1 superpermutation.
        # Wait, the blocks for n are the permutations of 1..n.
        # The "top-level" blocks for the n-th step are the permutations of 1..n-1.
        # Let's just build the word for n-1 and extract the windows of length n-1 that are permutations.
        
        # It's easier to just build the word for n-1.
        word = [1]
        for m in range(2, n_val + 1):
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

    # Get the word for n-1
    word_prev = get_blocks(n - 1)
    
    # Extract the blocks (permutations of 1..n-1) in order of appearance
    blocks = []
    seen = set()
    for i in range(len(word_prev) - (n - 1) + 1):
        window = tuple(word_prev[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            blocks.append(window)
            
    num_blocks = len(blocks)
    
    # 2. Compute transition costs
    # Cost C[i][j] is the number of new characters added when placing block j after block i.
    # Block i expands to: blocks[i] + [n] + blocks[i]
    # Block j expands to: blocks[j] + [n] + blocks[j]
    # We need the overlap between the suffix of the word ending with expansion of i
    # and the prefix of expansion of j.
    # Since the word ends with blocks[i] (the last part of the expansion),
    # we look for the longest suffix of (blocks[i]) that matches a prefix of (blocks[j] + [n] + blocks[j]).
    # Actually, the suffix of the *entire* previous word is just blocks[i] because the expansion is P, n, P.
    # So we check overlap between suffix of P and prefix of Q, n, Q.
    
    # Precompute expansions
    expansions = [list(b) + [n] + list(b) for b in blocks]
    
    # Cost matrix
    # C[i][j] = len(expansions[j]) - overlap(suffix of expansions[i], prefix of expansions[j])
    # But wait, the "suffix of expansions[i]" is just blocks[i].
    # Is it possible that the overlap extends further back?
    # In the standard construction, the overlap is calculated against the entire current word.
    # However, due to the structure, the maximum possible overlap is bounded by the length of the block.
    # Let's assume the overlap only depends on the last block for a first approximation.
    # If this heuristic is too weak, we might need to consider more context.
    # But for n=7, the blocks are length 6. The overlap is at most 6 or 7.
    
    C = [[0] * num_blocks for _ in range(num_blocks)]
    for i in range(num_blocks):
        exp_i = expansions[i]
        # Suffix of the word is exp_i (specifically the last part, which is blocks[i])
        # Actually, the word ends with blocks[i].
        suffix = blocks[i]
        for j in range(num_blocks):
            exp_j = expansions[j]
            # Find max overlap k such that suffix[-k:] == exp_j[:k]
            max_k = min(len(suffix), len(exp_j))
            overlap = 0
            for k in range(max_k, -1, -1):
                if suffix[-k:] == exp_j[:k]:
                    overlap = k
                    break
            C[i][j] = len(exp_j) - overlap
            
    # 3. Solve TSP Path
    # Start with greedy nearest neighbor
    # Then 2-opt
    
    # Greedy
    visited = [False] * num_blocks
    path = []
    current = 0
    visited[0] = True
    path.append(0)
    
    for _ in range(num_blocks - 1):
        best_next = -1
        best_cost = float('inf')
        for next_node in range(num_blocks):
            if not visited[next_node]:
                cost = C[current][next_node]
                if cost < best_cost:
                    best_cost = cost
                    best_next = next_node
        if best_next == -1:
            break
        visited[best_next] = True
        path.append(best_next)
        current = best_next
        
    # 2-opt improvement
    # Check deadline
    if time.time() > deadline - 10:
        # Just use greedy
        pass
    else:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(num_blocks - 1):
                for j in range(i + 1, num_blocks):
                    # Check if reversing path[i+1..j] improves cost
                    # Cost before: C[path[i]][path[i+1]] + C[path[j]][path[j+1]] (if j+1 exists)
                    # Cost after: C[path[i]][path[j]] + C[path[i+1]][path[j+1]] (if j+1 exists)
                    # Note: path is a path, not a cycle.
                    
                    # Let u = path[i], v = path[i+1], w = path[j], x = path[j+1] (if exists)
                    # If j == num_blocks-1, x is None (end of path)
                    
                    cost_before = C[path[i]][path[i+1]]
                    if j + 1 < num_blocks:
                        cost_before += C[path[j]][path[j+1]]
                    else:
                        cost_before += 0 # No outgoing cost from end? 
                        # Actually, the total length is sum of C[path[k]][path[k+1]] for k=0..N-2
                        # Plus the length of the first block?
                        # Let's define total length = len(expansions[path[0]]) + sum(C[path[k]][path[k+1]] for k in 0..N-2)
                        # So we only care about the internal edges.
                        
                    cost_after = C[path[i]][path[j]]
                    if j + 1 < num_blocks:
                        cost_after += C[path[i+1]][path[j+1]]
                    else:
                        cost_after += 0
                        
                    if cost_after < cost_before:
                        # Reverse path[i+1..j]
                        path[i+1:j+1] = reversed(path[i+1:j+1])
                        improved = True
                        if time.time() > deadline - 5:
                            break
                if time.time() > deadline - 5:
                    break
                    
    # 4. Construct the final word
    # Start with first block's expansion
    final_word = list(expansions[path[0]])
    for k in range(1, num_blocks):
        exp = expansions[path[k]]
        # Calculate overlap with current final_word
        # The overlap is determined by C[path[k-1]][path[k]]
        # C[i][j] = len(exp_j) - overlap
        # So overlap = len(exp_j) - C[path[k-1]][path[k]]
        overlap = len(exp) - C[path[k-1]][path[k]]
        final_word += exp[overlap:]
        
    return "".join(map(str, final_word))
