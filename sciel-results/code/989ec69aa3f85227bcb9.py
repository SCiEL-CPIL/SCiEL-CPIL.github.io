"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from itertools import permutations

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    # Build the graph:
    # Nodes: (n-1)-tuples of distinct symbols from {1..n}
    # Edge from u to v where v = u[1:] + (missing_symbol,)
    # Each edge corresponds to an n-permutation: u + (missing_symbol,)
    
    all_syms = set(range(1, n+1))
    
    def get_missing(node):
        # node is a tuple of n-1 distinct symbols
        # Return the one symbol not in node
        node_set = set(node)
        for s in all_syms:
            if s not in node_set:
                return s
        return None
    
    def get_next(node):
        s = get_missing(node)
        return node[1:] + (s,)
    
    def node_to_perm(node):
        s = get_missing(node)
        return node + (s,)
    
    # Find all cycles
    all_nodes = set(permutations(range(1, n+1), n-1))
    visited = set()
    cycles = []
    
    for node in all_nodes:
        if node not in visited:
            cycle = []
            cur = node
            while cur not in visited:
                visited.add(cur)
                cycle.append(cur)
                cur = get_next(cur)
            cycles.append(cycle)
    
    # For each cycle, compute the list of n-permutations
    cycle_perms = []
    for cycle in cycles:
        perms = [node_to_perm(node) for node in cycle]
        cycle_perms.append(perms)
    
    # Now we need to order the cycles to minimize total word length.
    # When concatenating cycle A ending with perm p and cycle B starting with perm q,
    # the overlap is the largest k such that p[n-k:] == q[:k].
    
    m = len(cycles)
    
    # Precompute overlaps between end of cycle i and start of cycle j
    # overlap[i][j] = overlap between last perm of cycle i and first perm of cycle j
    overlaps = [[0]*m for _ in range(m)]
    for i in range(m):
        end_perm = cycle_perms[i][-1]
        for j in range(m):
            if i == j:
                continue
            start_perm = cycle_perms[j][0]
            # Find max overlap
            max_ovl = 0
            for k in range(1, n):
                if end_perm[n-k:] == start_perm[:k]:
                    max_ovl = k
            overlaps[i][j] = max_ovl
    
    # Greedy: start with the cycle that has the best average outgoing overlap
    # or just start with cycle 0 and greedily pick next
    used = [False] * m
    order = []
    
    # Start with the cycle that maximizes the sum of overlaps to unused cycles
    best_start = 0
    best_score = -1
    for i in range(m):
        score = sum(overlaps[i][j] for j in range(m) if j != i)
        if score > best_score:
            best_score = score
            best_start = i
    
    order.append(best_start)
    used[best_start] = True
    current = best_start
    
    for _ in range(m - 1):
        best_next = -1
        best_ovl = -1
        for j in range(m):
            if not used[j]:
                if overlaps[current][j] > best_ovl:
                    best_ovl = overlaps[current][j]
                    best_next = j
        if best_next == -1:
            for j in range(m):
                if not used[j]:
                    best_next = j
                    break
        order.append(best_next)
        used[best_next] = True
        current = best_next
    
    # Build the word
    word = []
    for ci in order:
        perms = cycle_perms[ci]
        for perm in perms:
            if not word:
                word.extend(perm)
            else:
                # Find max overlap between end of word and start of perm
                max_ovl = 0
                # The word ends with the previous perm, so check overlap up to n-1
                for k in range(1, min(n, len(word)+1)):
                    if k > len(word):
                        break
                    if tuple(word[-k:]) == perm[:k]:
                        max_ovl = k
                word.extend(perm[max_ovl:])
    
    return "".join(map(str, word))

# EVOLVE-BLOCK-END
