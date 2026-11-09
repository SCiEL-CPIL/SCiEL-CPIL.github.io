"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 50.0  # safety margin before 60s kill (90s wall-clock)
    
    def elapsed():
        return time.time() - start_time
    
    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    
    # Index: number -> index in numbers list
    idx = {num: i for i, num in enumerate(numbers)}
    
    # Precompute conflict graph using bitmasks
    # conflict_mask[i] is a bitmask where bit j is set if numbers[i] and numbers[j] conflict
    # Two numbers conflict if lcm(a,b) <= 256, i.e., a*b/gcd(a,b) <= 256
    conflict_mask = [0] * n
    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Weight of selecting number i (use Fraction for exact arithmetic)
    weights = [Fraction(1, numbers[i]) for i in range(n)]
    
    # Convert to float for speed in comparisons, but keep Fraction for final score
    weights_f = [float(w) for w in weights]
    
    def score_from_indices(indices):
        s = Fraction(0, 1)
        for i in indices:
            s += weights[i]
        return s
    
    def is_valid(indices):
        # Check no two indices conflict
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a = indices[i]
                b = indices[j]
                if conflict_mask[a] & (1 << b):
                    return False
        return True
    
    # Bron-Kerbosch with pivot and coloring bound for Maximum Weight Clique
    # in the complement (compatibility) graph
    
    # Build compatibility adjacency (complement of conflict)
    # compat_mask[i] is a bitmask of nodes compatible with i (lcm > 256)
    compat_mask = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if not (conflict_mask[i] & (1 << j)):
                compat_mask[i] |= (1 << j)
    
    best_sum = Fraction(0, 1)
    best_clique = []
    
    # Greedy coloring bound: color the compatibility subgraph induced by P
    # A clique can have at most one vertex from each color class
    # Upper bound on clique weight = sum of max weights per color class
    
    def color_bound(P_list, current_sum):
        """
        P_list is a list of node indices.
        Returns an upper bound on the max weight of a clique in the subgraph induced by P_list.
        Uses greedy coloring on the compatibility graph induced by P_list.
        """
        if not P_list:
            return current_sum
        
        # Sort by weight descending for tighter coloring
        sorted_P = sorted(P_list, key=lambda x: weights_f[x], reverse=True)
        
        color_classes = []  # list of lists of nodes
        
        for v in sorted_P:
            placed = False
            # Try to put v in an existing color class
            # v can be in class c if v is NOT adjacent to any node in class c
            # (i.e., v is compatible with none of them in the compat graph)
            for c_idx, class_nodes in enumerate(color_classes):
                compatible_with_any = False
                for u in class_nodes:
                    if compat_mask[v] & (1 << u):
                        compatible_with_any = True
                        break
                if not compatible_with_any:
                    class_nodes.append(v)
                    placed = True
                    break
            
            if not placed:
                color_classes.append([v])
        
        # Calculate bound: sum of max weights per color class
        bound = current_sum
        for class_nodes in color_classes:
            max_w = max(weights_f[v] for v in class_nodes)
            bound += max_w
            
        return bound
    
    def bron_kerbosch(R, P_mask, X_mask):
        nonlocal best_sum, best_clique
        
        if elapsed() > DEADLINE - 3:
            return
        
        # If P is empty, we have a maximal clique
        if P_mask == 0:
            if R:
                current_sum = sum(weights[v] for v in R)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_clique = R[:]
            return
        
        # Get list of nodes in P for coloring
        P_list = []
        m = P_mask
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            P_list.append(j)
            m ^= lsb
        
        # Calculate coloring bound
        if R:
            current_sum = sum(weights[v] for v in R)
        else:
            current_sum = Fraction(0, 1)
        
        bound = color_bound(P_list, current_sum)
        if bound <= best_sum:
            return
        
        # Pivot selection: choose vertex u in P ∪ X with max degree in P
        # to minimize the number of branches
        pivot_mask = P_mask | X_mask
        if pivot_mask == 0:
            # No pivot, iterate over all P
            candidates = P_list
        else:
            # Find vertex with max degree in P
            best_deg = -1
            pivot = -1
            pm = pivot_mask
            while pm:
                lsb = pm & -pm
                u = lsb.bit_length() - 1
                # Degree of u in P
                deg = bin(compat_mask[u] & P_mask).count('1')
                if deg > best_deg:
                    best_deg = deg
                    pivot = u
                pm ^= lsb
            
            # Candidates are P \ N(pivot)
            # If pivot is in P, candidates = P & ~compat_mask[pivot]
            # If pivot is in X only, candidates = P
            if P_mask & (1 << pivot):
                candidates_mask = P_mask & ~compat_mask[pivot]
            else:
                candidates_mask = P_mask
            
            candidates = []
            cm = candidates_mask
            while cm:
                lsb = cm & -cm
                j = lsb.bit_length() - 1
                candidates.append(j)
                cm ^= lsb
        
        # Iterate over candidates
        for v in candidates:
            # Recurse with v added to R
            new_R = R + [v]
            # New P: nodes in P that are compatible with v
            new_P = P_mask & compat_mask[v]
            # New X: nodes in X that are compatible with v
            new_X = X_mask & compat_mask[v]
            
            bron_kerbosch(new_R, new_P, new_X)
            
            # Remove v from P, add to X
            P_mask &= ~(1 << v)
            X_mask |= (1 << v)
            
            # Check time
            if elapsed() > DEADLINE - 3:
                return
    
    # Run Bron-Kerbosch
    # Initial P is all nodes, X is empty
    initial_P = (1 << n) - 1
    bron_kerbosch([], initial_P, 0)
    
    # If Bron-Kerbosch didn't find anything or timed out, fall back to greedy
    if not best_clique:
        # Simple greedy fallback
        selected = [False] * n
        available = [True] * n
        result = []
        
        while True:
            if elapsed() > DEADLINE - 2:
                break
            best_i = -1
            best_gain = 0.0
            for i in range(n):
                if not available[i]:
                    continue
                gain = weights_f[i]
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
            
            if best_i < 0 or best_gain <= 0:
                break
            
            selected[best_i] = True
            result.append(best_i)
            m = conflict_mask[best_i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                available[j] = False
                m ^= lsb
            available[best_i] = False
        
        best_clique = result
    
    # Convert indices to numbers
    result = [numbers[i] for i in best_clique]
    result.sort()
    
    if not result:
        result = [2]
    
    return result
# EVOLVE-BLOCK-END
