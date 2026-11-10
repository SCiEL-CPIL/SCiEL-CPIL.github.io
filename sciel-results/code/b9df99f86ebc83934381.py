"""Find graphs with no regular induced subgraph on 6+ vertices."""
import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Precompute all 6-vertex subsets for n up to 22
    # For n=20: C(20,6) = 38760 subsets
    # For n=21: C(21,6) = 54264 subsets
    # For n=22: C(22,6) = 74613 subsets
    
    def is_regular_subgraph(adj_bits, subset, n):
        """Check if the induced subgraph on subset is regular.
        adj_bits[i] is a bitset where bit j is set if edge (i,j) exists."""
        k = len(subset)
        # Count degrees of each vertex in the subset
        degrees = [0] * k
        for idx_i, vi in enumerate(subset):
            bits = adj_bits[vi]
            for idx_j, vj in enumerate(subset):
                if idx_i != idx_j and (bits >> vj) & 1:
                    degrees[idx_i] += 1
        # Check if all degrees are equal
        d0 = degrees[0]
        for d in degrees[1:]:
            if d != d0:
                return False
        return True
    
    def count_violations(adj_bits, subsets, n):
        """Count number of 6-vertex subsets that induce a regular subgraph."""
        count = 0
        for subset in subsets:
            if is_regular_subgraph(adj_bits, subset, n):
                count += 1
        return count
    
    def get_edges_from_bits(adj_bits, n):
        """Convert adjacency bits to edge list."""
        edges = []
        for i in range(n):
            bits = adj_bits[i]
            for j in range(i+1, n):
                if (bits >> j) & 1:
                    edges.append([i, j])
        return edges
    
    def verify(adj_bits, n):
        """Verify no regular induced subgraph on 6+ vertices."""
        # Check all 6-vertex subsets
        for subset in itertools.combinations(range(n), 6):
            if is_regular_subgraph(adj_bits, subset, n):
                return False
        # Check larger subsets (7+): a regular subgraph on k>6 vertices
        # means all vertices have the same degree within the subset
        # But if a k-vertex induced subgraph is regular, it contains
        # many 6-vertex subsets. However, a 6-vertex subset of a regular
        # k-graph may NOT be regular. So we must check all sizes >= 6.
        for size in range(7, n+1):
            for subset in itertools.combinations(range(n), size):
                k = size
                degrees = [0] * k
                for idx_i, vi in enumerate(subset):
                    bits = adj_bits[vi]
                    for idx_j, vj in enumerate(subset):
                        if idx_i != idx_j and (bits >> vj) & 1:
                            degrees[idx_i] += 1
                d0 = degrees[0]
                regular = True
                for d in degrees[1:]:
                    if d != d0:
                        regular = False
                        break
                if regular:
                    return False
        return True
    
    # Try n from 22 down to 5
    for n in range(22, 4, -1):
        if time.time() > deadline - 30:
            break
            
        # Generate all 6-subsets
        subsets = list(itertools.combinations(range(n), 6))
        
        # Local search: start with random graph, flip edges to reduce violations
        best_found = None
        
        for restart in range(50):
            if time.time() > deadline - 20:
                break
            
            # Random initial graph
            adj_bits = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)
            
            # Local search
            current_violations = count_violations(adj_bits, subsets, n)
            
            if current_violations == 0:
                # Verify fully (including larger subsets)
                if verify(adj_bits, n):
                    best_found = adj_bits[:]
                    break
            
            # Try to reduce violations by flipping edges
            no_improve = 0
            max_no_improve = 200
            while current_violations > 0 and no_improve < max_no_improve:
                if time.time() > deadline - 15:
                    break
                
                # Pick a random edge to flip
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                
                # Flip edge (i, j)
                adj_bits[i] ^= (1 << j)
                adj_bits[j] ^= (1 << i)
                
                new_violations = count_violations(adj_bits, subsets, n)
                
                if new_violations < current_violations:
                    current_violations = new_violations
                    no_improve = 0
                else:
                    # Revert
                    adj_bits[i] ^= (1 << j)
                    adj_bits[j] ^= (1 << i)
                    no_improve += 1
                    # Occasionally accept worse moves (simulated annealing-like)
                    if random.random() < 0.05 and new_violations <= current_violations + 2:
                        # Keep the flip
                        adj_bits[i] ^= (1 << j)
                        adj_bits[j] ^= (1 << i)
                        current_violations = new_violations
                        no_improve = 0
            
            if current_violations == 0:
                if verify(adj_bits, n):
                    best_found = adj_bits[:]
                    break
        
        if best_found is not None:
            edges = get_edges_from_bits(best_found, n)
            return {'n': n, 'edges': edges}
    
    # Fallback: return a small valid graph
    # The empty graph on 5 vertices is always valid
    return {'n': 5, 'edges': []}
