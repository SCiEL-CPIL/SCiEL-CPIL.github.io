import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    def verify(n, edges):
        """
        Verify the graph is C4-free and has min degree >= n-39.
        """
        if n < 1 or n > 47:
            return False
        target_deg = max(0, n - 39)
        
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            if u < 0 or u >= n or v < 0 or v >= n or u == v:
                return False
            adj[u, v] = 1
            adj[v, u] = 1
        
        degrees = adj.sum(axis=1)
        if np.min(degrees) < target_deg:
            return False
            
        # Check C4-free: A2[i,j] >= 2 implies C4
        A = adj.astype(np.int32)
        A2 = A @ A
        np.fill_diagonal(A2, 0)
        if np.any(A2 >= 2):
            return False
            
        return True

    def simulated_annealing(n, target_deg, time_limit):
        """
        Simulated Annealing to find a C4-free graph with min degree >= target_deg.
        """
        sa_start = time.time()
        
        # Initialize adjacency matrix and neighbor sets
        # Start with a random graph of moderate density
        p_init = target_deg / (n - 1) * 1.5
        adj = np.zeros((n, n), dtype=np.int8)
        neighbors = [set() for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p_init:
                    adj[i, j] = 1
                    adj[j, i] = 1
                    neighbors[i].add(j)
                    neighbors[j].add(i)
                    
        def get_energy():
            """
            Energy function:
            1. Penalty for C4s: For each pair (u,v), if they have >= 2 common neighbors, 
               that's a C4 violation. We count the number of such pairs (or sum of excess common neighbors).
               Actually, a single pair with k common neighbors implies k choose 2 C4s.
               But for the SA move, we just need a scalar.
               Let's use: sum over all pairs (u,v) of max(0, common(u,v) - 1).
               This penalizes any pair having >1 common neighbor.
               
            2. Penalty for low degree: sum over all vertices of max(0, target_deg - deg(v)).
            
            We want to minimize Energy.
            """
            c4_penalty = 0
            # Efficiently compute common neighbors count?
            # A2 = A @ A. A2[i,j] is number of common neighbors.
            # We need sum(max(0, A2[i,j] - 1)) for i != j.
            # Computing A @ A is O(n^3) ~ 47^3 ~ 100k ops. Fast enough for 150s?
            # 150s / 0.001s per step = 150k steps.
            # But we can update incrementally?
            # For now, let's compute it directly but maybe optimize if slow.
            # Actually, n=47 is small. 47^3 = 103823.
            # Python loops are slow. NumPy is fast.
            A = adj.astype(np.int32)
            A2 = A @ A
            np.fill_diagonal(A2, 0)
            
            # Penalty for pairs with >= 2 common neighbors
            # We want to count how many pairs violate the C4-free condition.
            # A C4-free graph requires A2[i,j] <= 1 for all i != j.
            # So penalty = sum(A2[i,j] - 1 for A2[i,j] > 1)
            excess = np.maximum(A2 - 1, 0)
            c4_penalty = np.sum(excess)
            
            # Degree penalty
            degrees = adj.sum(axis=1)
            deg_deficit = np.maximum(target_deg - degrees, 0)
            deg_penalty = np.sum(deg_deficit)
            
            # Weight the C4 penalty heavily to prioritize C4-freeness
            # If we have a C4, we don't care about degree yet.
            return 100 * c4_penalty + deg_penalty

        current_energy = get_energy()
        best_energy = current_energy
        best_adj = adj.copy()
        
        T = 1.0
        cooling_rate = 0.999
        
        while time.time() - sa_start < time_limit:
            # Pick a random move: add or remove an edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
                
            # Ensure u < v for consistent state? No, adj is symmetric.
            # But for neighbor sets, we need to be careful.
            
            old_energy = current_energy
            
            # Perform move
            if adj[u, v] == 1:
                # Remove edge
                adj[u, v] = 0
                adj[v, u] = 0
                neighbors[u].remove(v)
                neighbors[v].remove(u)
            else:
                # Add edge
                adj[u, v] = 1
                adj[v, u] = 1
                neighbors[u].add(v)
                neighbors[v].add(u)
                
            new_energy = get_energy()
            delta = new_energy - old_energy
            
            # Accept if better, or with probability exp(-delta / T)
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_adj = adj.copy()
            else:
                # Revert
                if adj[u, v] == 1:
                    adj[u, v] = 0
                    adj[v, u] = 0
                    neighbors[u].remove(v)
                    neighbors[v].remove(u)
                else:
                    adj[u, v] = 1
                    adj[v, u] = 1
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    
            T *= cooling_rate
            if T < 0.01:
                T = 0.01
            
            # Early exit if perfect solution found
            if best_energy == 0:
                break

        # Check if best solution is valid
        # best_energy == 0 implies c4_penalty=0 and deg_penalty=0
        if best_energy == 0:
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if best_adj[i, j] == 1]
            return edges
        return None

    # Main Logic
    # Try n=47, 46, etc.
    # For n=47, target_deg = 8.
    # For n=46, target_deg = 7.
    
    # We will try multiple random restarts for each n.
    
    for n in [47, 46, 45, 44]:
        if time.time() - start_time > deadline - 5:
            break
            
        target_deg = max(0, n - 39)
        
        # Try a few times
        for attempt in range(10):
            if time.time() - start_time > deadline - 2:
                break
                
            # Run SA
            # Give it a chunk of time
            sa_time_limit = min(10, deadline - (time.time() - start_time) - 2)
            if sa_time_limit <= 0:
                break
                
            edges = simulated_annealing(n, target_deg, sa_time_limit)
            
            if edges is not None:
                # Verify
                if verify(n, edges):
                    return {'n': n, 'edges': edges}
                    
    # Fallback
    return {'n': 39, 'edges': []}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
