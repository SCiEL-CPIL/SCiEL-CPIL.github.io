import time
import random
import math
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # 1. Precompute GCDs and Conflicts
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # conflicts[v] is a set of frozensets {u, w} such that {v, u, w} is a bad triple
    conflicts = [set() for _ in range(N + 1)]
    
    for a, b, c in combinations(numbers, 3):
        g_ab = gcd_table[a][b]
        g_ac = gcd_table[a][c]
        g_bc = gcd_table[b][c]
        if g_ab == g_ac == g_bc:
            conflicts[a].add(frozenset((b, c)))
            conflicts[b].add(frozenset((a, c)))
            conflicts[c].add(frozenset((a, b)))

    # Precompute degrees for greedy sorting
    degrees = [len(conflicts[v]) for v in range(N + 1)]
    
    # Convert to 0-indexed: element i (1-indexed) -> index i-1
    # We'll work with indices 0..127 representing numbers 1..128
    
    # Build conflict structure using 0-indexed representation
    # conflict_pairs[v_idx] = set of frozensets of (u_idx, w_idx)
    conflict_pairs = [set() for _ in range(N)]
    for v in range(1, N + 1):
        for p in conflicts[v]:
            u, w = tuple(p)
            # Convert to 0-indexed
            u_idx, w_idx = u - 1, w - 1
            conflict_pairs[v - 1].add(frozenset((u_idx, w_idx)))
    
    # Precompute all bad triples as frozensets for quick lookup
    # bad_triples is a set of frozensets of 3 indices (0-indexed)
    bad_triples = set()
    for a_idx in range(N):
        for b_idx in range(a_idx + 1, N):
            for c_idx in range(b_idx + 1, N):
                a, b, c = a_idx + 1, b_idx + 1, c_idx + 1
                g_ab = gcd_table[a][b]
                g_ac = gcd_table[a][c]
                g_bc = gcd_table[b][c]
                if g_ab == g_ac == g_bc:
                    bad_triples.add(frozenset((a_idx, b_idx, c_idx)))
    
    # Simulated Annealing
    # State: x = list of booleans, x[i] = True if number (i+1) is in the set
    # Energy: penalty for each bad triple that is fully included
    # We want to maximize |S| subject to no bad triple being fully in S
    
    # Initial solution: greedy by degree
    def greedy_init():
        x = [False] * N
        # Sort by degree (ascending)
        order = sorted(range(N), key=lambda i: degrees[i + 1])
        for v_idx in order:
            # Check if adding v_idx creates any bad triple
            can_add = True
            for p in conflict_pairs[v_idx]:
                u_idx, w_idx = tuple(p)
                if x[u_idx] and x[w_idx]:
                    can_add = False
                    break
            if can_add:
                x[v_idx] = True
        return x
    
    def compute_energy(x):
        """Compute the number of bad triples fully included in x."""
        energy = 0
        for triple in bad_triples:
            a, b, c = tuple(triple)
            if x[a] and x[b] and x[c]:
                energy += 1
        return energy
    
    def compute_size(x):
        return sum(x)
    
    best_x = None
    best_size = 0
    best_energy = float('inf')
    
    # Run simulated annealing
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize
        if restart == 0:
            x = greedy_init()
        else:
            # Random initialization with some structure
            x = [False] * N
            # Start with a random subset of moderate size
            k = random.randint(40, 70)
            indices = random.sample(range(N), k)
            for i in indices:
                x[i] = True
            # Quick local cleanup: remove elements that create bad triples
            changed = True
            while changed:
                changed = False
                for i in range(N):
                    if x[i]:
                        # Check if removing i reduces energy
                        for p in conflict_pairs[i]:
                            u, w = tuple(p)
                            if x[u] and x[w]:
                                x[i] = False
                                changed = True
                                break
        
        current_energy = compute_energy(x)
        current_size = compute_size(x)
        
        if current_energy == 0 and current_size > best_size:
            best_size = current_size
            best_x = list(x)
        
        # Simulated annealing
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        steps_per_temp = 200
        
        while T > T_min and time.time() < deadline:
            for _ in range(steps_per_temp):
                if time.time() > deadline:
                    break
                
                # Random move: flip a random bit
                i = random.randint(0, N - 1)
                old_val = x[i]
                x[i] = not old_val
                
                # Compute delta energy
                # If we added i (old_val=False, new=True):
                #   Delta = number of bad triples containing i where the other two are in x
                # If we removed i (old_val=True, new=False):
                #   Delta = -number of bad triples containing i where the other two are in x
                if not old_val:
                    # Added i
                    delta_e = 0
                    for p in conflict_pairs[i]:
                        u, w = tuple(p)
                        if x[u] and x[w]:
                            delta_e += 1
                else:
                    # Removed i
                    delta_e = 0
                    for p in conflict_pairs[i]:
                        u, w = tuple(p)
                        if x[u] and x[w]:
                            delta_e += 1
                    delta_e = -delta_e
                
                # Acceptance probability
                if delta_e <= 0:
                    accept = True
                else:
                    # delta_e > 0, accept with probability exp(-delta_e / T)
                    accept = random.random() < math.exp(-delta_e / T)
                
                if not accept:
                    # Revert
                    x[i] = old_val
                else:
                    current_energy += delta_e
                    current_size += (1 if not old_val else -1)
                    
                    # Update best if valid
                    if current_energy == 0 and current_size > best_size:
                        best_size = current_size
                        best_x = list(x)
            
            T *= cooling
        
        # Local search: try to add more elements
        if time.time() < deadline:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(N):
                    if not x[i]:
                        # Check if we can add i
                        can_add = True
                        for p in conflict_pairs[i]:
                            u, w = tuple(p)
                            if x[u] and x[w]:
                                can_add = False
                                break
                        if can_add:
                            x[i] = True
                            current_size += 1
                            current_energy = compute_energy(x)
                            improved = True
                            if current_energy == 0 and current_size > best_size:
                                best_size = current_size
                                best_x = list(x)
                            break
            
            # Try 1-1 swaps
            if time.time() < deadline:
                for i in range(N):
                    if x[i]:
                        x[i] = False
                        current_size -= 1
                        for j in range(N):
                            if not x[j]:
                                # Check if we can add j
                                can_add = True
                                for p in conflict_pairs[j]:
                                    u, w = tuple(p)
                                    if x[u] and x[w]:
                                        can_add = False
                                        break
                                if can_add:
                                    x[j] = True
                                    current_size += 1
                                    current_energy = compute_energy(x)
                                    if current_energy == 0 and current_size > best_size:
                                        best_size = current_size
                                        best_x = list(x)
                                    break
                        x[i] = True
                        current_size += 1
    
    if best_x is None:
        # Fallback: return a valid small set
        return [1]
    
    return [i + 1 for i in range(N) if best_x[i]]
