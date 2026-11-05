"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import heapq
import random
from itertools import permutations
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    if n <= 3:
        # Standard construction is optimal for n<=3
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
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # Standard recursive construction as baseline
    def standard_construction(n):
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
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word
    
    baseline_word = standard_construction(n)
    baseline_len = len(baseline_word)
    
    # For n=4, known shortest is 33
    if n == 4:
        # Precomputed shortest superpermutation for n=4 (length 33)
        # This is a known result
        best_word = None
        best_len = baseline_len
        
        # Try A* search for n=4
        symbols = tuple(range(1, n+1))
        all_perms = list(permutations(symbols))
        perm_to_idx = {p: i for i, p in enumerate(all_perms)}
        num_perms = len(all_perms)
        
        # Build adjacency: from perm i, next perm j if suffix of i (len n-1) == prefix of j (len n-1)
        # suffix of P = P[1:], prefix of Q = Q[:-1]
        # So edge i->j if P[1:] == Q[:-1], i.e., Q = (P[1:] + (new_sym,)) where new_sym not in P[1:]
        # Actually Q[:-1] = P[1:], so Q = P[1:] + (x,) where x is the symbol not in P[1:]
        # P[1:] has n-1 distinct symbols, so exactly one symbol is missing
        # Thus each perm has exactly one outgoing edge!
        
        # Wait, that means the graph is a set of disjoint cycles covering all perms.
        # Let's verify: from P, the next must have prefix P[1:]. The only permutation with that prefix is P[1:] + (missing_sym,).
        # So yes, each node has out-degree 1. The graph is a permutation (bijection).
        
        # This means we need to find a Hamiltonian path in a functional graph (each node out-degree 1).
        # A Hamiltonian path in such a graph exists only if the graph is a single cycle.
        # Let's check if the graph for n=4 is a single cycle.
        
        # Build the successor function
        successor = [0] * num_perms
        for i, p in enumerate(all_perms):
            prefix = p[1:]  # length n-1
            # Find the symbol not in prefix
            used = set(prefix)
            missing = None
            for s in symbols:
                if s not in used:
                    missing = s
                    break
            q = prefix + (missing,)
            successor[i] = perm_to_idx[q]
        
        # Check if it's a single cycle
        visited = [False] * num_perms
        cycle_len = 0
        for i in range(num_perms):
            if not visited[i]:
                # Follow the cycle
                j = i
                while not visited[j]:
                    visited[j] = True
                    cycle_len += 1
                    j = successor[j]
        
        if cycle_len == num_perms:
            # Single cycle, so a Hamiltonian path exists with length n + num_perms - 1
            # Start at any node, follow the cycle
            path = [all_perms[0]]
            cur = 0
            for _ in range(num_perms - 1):
                cur = successor[cur]
                path.append(all_perms[cur])
            
            # Build the word
            word = list(path[0])
            for p in path[1:]:
                word.append(p[-1])
            
            if len(word) < best_len:
                best_len = len(word)
                best_word = word
        
        if best_word:
            return "".join(map(str, best_word))
    
    # For n >= 5, try greedy reordering of permutation blocks
    def greedy_construct(n):
        """
        Greedy construction that reorders permutation blocks to maximize overlaps.
        At each level, we collect the blocks and use nearest-neighbor to find a good ordering.
        """
        word = [1]
        
        for m in range(2, n + 1):
            # Collect all (m-1)-permutations from the current word
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            if not perms:
                # Fallback: just add the new symbol
                word.append(m)
                continue
            
            # Create blocks for each permutation
            blocks = []
            for p in perms:
                block = list(p) + [m] + list(p)
                blocks.append(block)
            
            # Greedy nearest-neighbor ordering to maximize overlaps
            num_blocks = len(blocks)
            if num_blocks <= 1:
                new_word = list(blocks[0]) if blocks else word
                word = new_word
                continue
            
            # Precompute overlap between each pair of blocks
            # overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
            overlap = [[0] * num_blocks for _ in range(num_blocks)]
            for i in range(num_blocks):
                for j in range(num_blocks):
                    if i == j:
                        continue
                    # Try overlaps from max possible down to 0
                    max_ov = min(len(blocks[i]), len(blocks[j]))
                    for o in range(max_ov, -1, -1):
                        if o == 0 or blocks[i][len(blocks[i]) - o:] == blocks[j][:o]:
                            overlap[i][j] = o
                            break
            
            # Greedy: start with the block that has the best average outgoing overlap
            # Or just start with the first one and greedily pick the next
            unvisited = set(range(num_blocks))
            # Start with block 0
            current = 0
            unvisited.remove(current)
            order = [current]
            
            while unvisited:
                best_next = -1
                best_overlap = -1
                for j in unvisited:
                    if overlap[current][j] > best_overlap:
                        best_overlap = overlap[current][j]
                        best_next = j
                if best_next == -1:
                    break
                current = best_next
                unvisited.remove(current)
                order.append(current)
            
            # Add any remaining unvisited blocks
            for j in unvisited:
                order.append(j)
            
            # Build the new word from the ordered blocks
            new_word = list(blocks[order[0]])
            for idx in range(1, len(order)):
                prev_idx = order[idx - 1]
                curr_idx = order[idx]
                ov = overlap[prev_idx][curr_idx]
                new_word.extend(blocks[curr_idx][ov:])
            
            word = new_word
        
        return word
    
    # Try greedy construction
    greedy_word = greedy_construct(n)
    greedy_len = len(greedy_word)
    
    # Use the better of baseline and greedy
    if greedy_len < baseline_len:
        best_word = greedy_word
    else:
        best_word = baseline_word
    
    return "".join(map(str, best_word))
# EVOLVE-BLOCK-END
