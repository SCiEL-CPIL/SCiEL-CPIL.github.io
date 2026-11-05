import time
import random
import numpy as np
from itertools import combinations

def _edges_from_adj_masks(adj_masks, n):
    edges = []
    for i in range(n):
        m = adj_masks[i]
        for j in range(i + 1, n):
            if (m >> j) & 1:
                edges.append([i, j])
    return edges

def _count_violations(adj_masks, n, max_k=None):
    """
    Counts the number of regular induced subgraphs of size >= 6.
    For speed, we might limit max_k.
    """
    if max_k is None:
        max_k = n
        
    count = 0
    # Check sizes from 6 to max_k
    for k in range(6, max_k + 1):
        # Generate all subsets of size k
        # Using itertools.combinations
        for subset in combinations(range(n), k):
            # Check if regular
            # Get degrees of vertices in subset
            degrees = []
            for v in subset:
                # Count neighbors in subset
                deg = 0
                m = adj_masks[v]
                # Mask for subset
                # Instead of building mask every time, we can just check bits
                # But subset is a tuple.
                # Faster: precompute mask for subset?
                # Or just iterate subset again?
                # k is small (6-10).
                for u in subset:
                    if u != v and ((adj_masks[v] >> u) & 1):
                        deg += 1
                degrees.append(deg)
            
            # Check if all degrees are same
            if len(set(degrees)) == 1:
                count += 1
                # Early exit for search purposes? 
                # If we just need to know if >0, we can return 1.
                # But for SA, we want the count.
                # However, if count is high, we don't need exact number.
                # Let's cap the count to avoid long runs.
                if count > 100:
                    return count
    
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_n = 0
    best_adj_masks = []
    best_energy = float('inf')
    
    # Try n=22, 21, 20
    for target_n in [22, 21, 20]:
        if time.time() > deadline - 10:
            break
            
        n = target_n
        
        # Simulated Annealing
        # Initialize
        adj_masks = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
        
        current_energy = _count_violations(adj_masks, n, max_k=8) # Limit k for speed
        
        T = 1.0
        min_T = 0.01
        cooling_rate = 0.995
        
        while time.time() < deadline - 5:
            if current_energy == 0:
                # Found valid graph
                best_n = n
                best_adj_masks = adj_masks
                best_energy = 0
                break
            
            # Pick random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
                
            # Flip edge
            if (adj_masks[u] >> v) & 1:
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
            else:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                
            new_energy = _count_violations(adj_masks, n, max_k=8)
            
            if new_energy < current_energy:
                current_energy = new_energy
            else:
                # Accept with probability
                delta = new_energy - current_energy
                if delta < 0:
                    accept = True
                else:
                    # Boltzmann factor
                    # Use random
                    if random.random() < np.exp(-delta / T):
                        accept = True
                    else:
                        accept = False
                        
                if not accept:
                    # Revert flip
                    if (adj_masks[u] >> v) & 1:
                        adj_masks[u] &= ~(1 << v)
                        adj_masks[v] &= ~(1 << u)
                    else:
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)
                        
            T *= cooling_rate
            if T < min_T:
                T = min_T
                
        if best_n >= n:
            break

    if best_n == 0:
        # Fallback
        best_n = 5
        best_adj_masks = [0] * 5

    edges = _edges_from_adj_masks(best_adj_masks, best_n)
    return {'n': best_n, 'edges': edges}
