import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110  # safety margin
    
    n = 20
    # Bitset representation: N[i] is a bitmask of neighbors of vertex i
    N = [0] * n
    
    def set_edge(i, j, present):
        if present:
            N[i] |= (1 << j)
            N[j] |= (1 << i)
        else:
            N[i] &= ~(1 << j)
            N[j] &= ~(1 << i)
    
    def count_violations():
        """Count number of triples with >= 3 common neighbors."""
        count = 0
        for a in range(n):
            for b in range(a+1, n):
                ab = N[a] & N[b]
                for c in range(b+1, n):
                    abc = ab & N[c]
                    if abc & (abc - 1):  # at least 2 bits? No, we need >= 3
                        # Check if popcount >= 3
                        if bin(abc).count('1') >= 3:
                            count += 1
        return count
    
    # Actually, let me be more careful. We need >= 3 common neighbors.
    # popcount >= 3 means at least 3 bits set.
    
    def popcount(x):
        return bin(x).count('1')
    
    def count_violations2():
        count = 0
        for a in range(n):
            for b in range(a+1, n):
                ab = N[a] & N[b]
                for c in range(b+1, n):
                    abc = ab & N[c]
                    if popcount(abc) >= 3:
                        count += 1
        return count
    
    # Greedy construction: start empty, add edges that don't create violation
    # Better: start with all edges, remove greedily
    # Even better: random initialization + SA
    
    # Let's do: start with a random graph, then SA to minimize violations,
    # then among zero-violation states, maximize edges.
    
    # Actually, let's use a two-phase approach:
    # Phase 1: SA to find a K3,3-free graph with many edges
    # State: which edges are present
    # Move: toggle a random edge
    # Energy: if violations > 0, energy = violations; else energy = -edges
    
    # Better energy: energy = violations * 1000 - edge_count
    # We minimize this.
    
    # Let's use numpy for speed in some places, but the bitset approach is already fast.
    
    # Initialize: random graph with moderate density
    random.seed(42)
    
    # Start with a greedy construction for a good initial solution
    N = [0] * n
    # Try all edges in random order, add if no violation
    edges_all = [(i,j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(edges_all)
    
    def is_valid_add(i, j):
        """Check if adding edge (i,j) keeps the graph K3,3-free."""
        # After adding (i,j), check all triples involving i or j
        # A triple (a,b,c) has common neighbors including both i and j only if
        # i and j are both neighbors of all of a,b,c
        # So we need: for no triple (a,b,c) distinct from i,j are both i,j in N[a]&N[b]&N[c]
        # i.e., N[a]&N[b]&N[c] contains both bit i and bit j
        # Equivalently, (N[a]&N[b]&N[c]) & ((1<<i)|(1<<j)) == ((1<<i)|(1<<j))
        mask_ij = (1 << i) | (1 << j)
        for a in range(n):
            if a == i or a == j:
                continue
            for b in range(a+1, n):
                if b == i or b == j:
                    continue
                for c in range(b+1, n):
                    if c == i or c == j:
                        continue
                    common = N[a] & N[b] & N[c]
                    if (common & mask_ij) == mask_ij:
                        return False
        return True
    
    # This is O(n^3) per edge check, too slow for greedy.
    # Let's just do SA directly.
    
    # SA approach:
    # State: N (list of 20 bitmasks)
    # Initialize randomly
    N = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                N[i] |= (1 << j)
                N[j] |= (1 << i)
    
    def edge_count():
        total = 0
        for i in range(n):
            total += popcount(N[i])
        return total // 2
    
    def energy():
        v = count_violations2()
        e = edge_count()
        return v * 10000 - e
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.01
    cooling = 0.999
    
    best_N = list(N)
    best_e = edge_count()
    best_v = count_violations2()
    if best_v == 0:
        pass  # already valid
    
    current_e = edge_count()
    current_v = count_violations2()
    current_energy = current_v * 10000 - current_e
    
    step = 0
    while time.time() < deadline:
        # Pick random edge to toggle
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        # Compute delta energy
        # Toggle edge (i,j)
        had_edge = (N[i] >> j) & 1
        
        # Apply toggle
        if had_edge:
            N[i] &= ~(1 << j)
            N[j] &= ~(1 << i)
        else:
            N[i] |= (1 << j)
            N[j] |= (1 << i)
        
        new_v = count_violations2()
        new_e = edge_count()
        new_energy = new_v * 10000 - new_e
        
        delta = new_energy - current_energy
        
        if delta < 0 or random.random() < (2.718281828 ** (-delta / T)):
            current_energy = new_energy
            current_v = new_v
            current_e = new_e
            if new_v == 0 and new_e > best_e:
                best_e = new_e
                best_N = list(N)
                best_v = 0
        else:
            # Revert
            if had_edge:
                N[i] |= (1 << j)
                N[j] |= (1 << i)
            else:
                N[i] &= ~(1 << j)
                N[j] &= ~(1 << i)
        
        T *= cooling
        if T < T_min:
            T = T_min
        
        step += 1
        if step % 10000 == 0:
            if time.time() > deadline - 5:
                break
    
    # Return best valid graph
    if best_v == 0:
        result = []
        for i in range(n):
            for j in range(i+1, n):
                if (best_N[i] >> j) & 1:
                    result.append([i, j])
        return result
    else:
        # Fallback: return the seed
        return [[a, b] for a in range(2) for b in range(2, 20)]
