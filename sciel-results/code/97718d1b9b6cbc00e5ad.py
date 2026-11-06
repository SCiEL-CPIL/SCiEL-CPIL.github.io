import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    n = 20
    # We'll work with a bipartite graph: left = 0..9, right = 10..19
    left = list(range(10))
    right = list(range(10, 20))
    
    def elapsed():
        return time.time() - start_time
    
    def check_k33(adj_lr, adj_rl):
        """Check if bipartite graph has K3,3. Returns list of (L3, R3) violations."""
        violations = []
        for L3 in itertools.combinations(left, 3):
            # Find common neighbors in right
            common = set(adj_lr[L3[0]]) & set(adj_lr[L3[1]]) & set(adj_lr[L3[2]])
            if len(common) >= 3:
                for R3 in itertools.combinations(common, 3):
                    violations.append((L3, R3))
        return violations
    
    def count_violations(adj_lr):
        """Count total K3,3 subgraphs."""
        count = 0
        for L3 in itertools.combinations(left, 3):
            common = set(adj_lr[L3[0]]) & set(adj_lr[L3[1]]) & set(adj_lr[L3[2]])
            if len(common) >= 3:
                count += len(list(itertools.combinations(common, 3)))
        return count
    
    def is_valid_bipartite(adj_lr):
        """Check if no K3,3 exists."""
        for L3 in itertools.combinations(left, 3):
            common = set(adj_lr[L3[0]]) & set(adj_lr[L3[1]]) & set(adj_lr[L3[2]])
            if len(common) >= 3:
                return False
        return True
    
    def edge_count(adj_lr):
        return sum(len(neigh) for neigh in adj_lr.values())
    
    def try_add_edge(adj_lr, u, v):
        """Try to add edge u-v, return True if valid."""
        if v in adj_lr[u]:
            return False
        # Check if adding this edge creates a K3,3
        # For each pair of other left vertices that both connect to v,
        # check if u and those two have >= 3 common neighbors including v
        adj_lr[u].add(v)
        valid = True
        # Check all triples containing u in left
        for other1 in left:
            if other1 == u:
                continue
            if v not in adj_lr[other1]:
                continue
            for other2 in left:
                if other2 == u or other2 == other1:
                    continue
                if v not in adj_lr[other2]:
                    continue
                common = set(adj_lr[u]) & set(adj_lr[other1]) & set(adj_lr[other2])
                if len(common) >= 3:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            adj_lr[u].remove(v)
        return valid
    
    def local_search(adj_lr, max_iter=5000):
        """Local search: try to add edges, remove problematic ones."""
        best_adj = {u: set(adj_lr[u].copy()) for u in left}
        best_count = edge_count(best_adj)
        
        for _ in range(max_iter):
            if elapsed() > DEADLINE:
                break
            
            improved = False
            
            # Try to add each missing edge
            for u in left:
                for v in right:
                    if v not in adj_lr[u]:
                        if try_add_edge(adj_lr, u, v):
                            improved = True
                            if edge_count(adj_lr) > best_count:
                                best_adj = {x: set(adj_lr[x].copy()) for x in left}
                                best_count = edge_count(best_adj)
            
            if not improved:
                break
        
        return best_adj, best_count
    
    def greedy_remove(adj_lr):
        """Start from complete bipartite, greedily remove edges to eliminate K3,3."""
        adj = {u: set(right.copy()) for u in left}
        
        for iteration in range(1000):
            if elapsed() > DEADLINE:
                break
            
            if is_valid_bipartite(adj):
                break
            
            # Find all K3,3 violations
            violations = []
            for L3 in itertools.combinations(left, 3):
                common = set(adj[L3[0]]) & set(adj[L3[1]]) & set(adj[L3[2]])
                if len(common) >= 3:
                    for R3 in itertools.combinations(common, 3):
                        violations.append((L3, R3))
            
            if not violations:
                break
            
            # Count how many violations each edge appears in
            edge_violation_count = defaultdict(int)
            for L3, R3 in violations:
                for u in L3:
                    for v in R3:
                        edge_violation_count[(u, v)] += 1
            
            # Remove the edge that appears in the most violations
            if edge_violation_count:
                worst_edge = max(edge_violation_count, key=lambda e: edge_violation_count[e])
                u, v = worst_edge
                if v in adj[u]:
                    adj[u].remove(v)
        
        return adj
    
    # Strategy 1: Greedy removal from complete bipartite
    best_edges = []
    best_count = 0
    
    # Run greedy removal
    adj = greedy_remove({u: set(right.copy()) for u in left})
    count = edge_count(adj)
    if count > best_count:
        best_count = count
        best_edges = [[u, v] for u in left for v in adj[u]]
    
    # Strategy 2: Local search from greedy result
    if elapsed() < DEADLINE:
        adj_ls, count_ls = local_search(adj, max_iter=3000)
        if count_ls > best_count:
            best_count = count_ls
            best_edges = [[u, v] for u in left for v in adj_ls[u]]
    
    # Strategy 3: Multiple random initializations
    for trial in range(20):
        if elapsed() > DEADLINE:
            break
        
        # Random dense bipartite graph
        adj_rand = {u: set() for u in left}
        for u in left:
            for v in right:
                if random.random() < 0.7:
                    adj_rand[u].add(v)
        
        # Local search from random start
        adj_res, count_res = local_search(adj_rand, max_iter=2000)
        if count_res > best_count:
            best_count = count_res
            best_edges = [[u, v] for u in left for v in adj_res[u]]
    
    # Strategy 4: Try different random seeds for greedy with perturbation
    for trial in range(10):
        if elapsed() > DEADLINE:
            break
        
        # Start from complete, but remove some random edges first
        adj_pert = {u: set(right.copy()) for u in left}
        # Remove 5-15 random edges
        num_remove = random.randint(5, 15)
        all_edges = [(u, v) for u in left for v in right]
        random.shuffle(all_edges)
        for u, v in all_edges[:num_remove]:
            if v in adj_pert[u]:
                adj_pert[u].remove(v)
        
        # Greedy removal
        adj_pert = greedy_remove(adj_pert)
        count_pert = edge_count(adj_pert)
        
        # Local search
        adj_ls2, count_ls2 = local_search(adj_pert, max_iter=2000)
        
        if count_ls2 > best_count:
            best_count = count_ls2
            best_edges = [[u, v] for u in left for v in adj_ls2[u]]
    
    # Strategy 5: Simulated annealing style - try removing and re-adding
    if elapsed() < DEADLINE:
        adj_sa = {u: set(adj[u].copy()) for u in left}
        best_adj_sa = {u: set(adj[u].copy()) for u in left}
        best_count_sa = edge_count(best_adj_sa)
        
        for sa_iter in range(10000):
            if elapsed() > DEADLINE:
                break
            
            # Randomly remove an edge
            u = random.choice(left)
            if adj_sa[u]:
                v = random.choice(list(adj_sa[u]))
                adj_sa[u].remove(v)
            
            # Try to add random missing edges
            for _ in range(5):
                u = random.choice(left)
                missing = [v for v in right if v not in adj_sa[u]]
                if missing:
                    v = random.choice(missing)
                    if try_add_edge(adj_sa, u, v):
                        pass
            
            count_sa = edge_count(adj_sa)
            if count_sa > best_count_sa:
                best_count_sa = count_sa
                best_adj_sa = {x: set(adj_sa[x].copy()) for x in left}
        
        if best_count_sa > best_count:
            best_count = best_count_sa
            best_edges = [[u, v] for u in left for v in best_adj_sa[u]]
    
    # Ensure we return a valid result
    if not best_edges:
        # Fallback: K2,18
        best_edges = [[a, b] for a in range(2) for b in range(2, 20)]
    
    return best_edges

# EVOLVE-BLOCK-END
