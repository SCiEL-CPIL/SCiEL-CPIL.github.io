import random
import time
from itertools import combinations

def construct():
    deadline = time.time() + 110  # safety margin
    n = 20
    
    # Adjacency as set of frozensets for fast lookup
    # Use a set of (u,v) with u < v for undirected edges
    
    def make_edge_set(edge_list):
        s = set()
        for u, v in edge_list:
            if u > v:
                u, v = v, u
            s.add((u, v))
        return s
    
    def has_k33(adj):
        """Check if graph has K3,3. Returns True if violation exists."""
        # For each 3-set of vertices, check if they have >= 3 common neighbours
        # Common neighbours of {a,b,c} = N(a) ∩ N(b) ∩ N(c)
        # Build neighbour sets
        nbrs = [set() for _ in range(n)]
        for (u, v) in adj:
            nbrs[u].add(v)
            nbrs[v].add(u)
        for a, b, c in combinations(range(n), 3):
            common = nbrs[a] & nbrs[b] & nbrs[c]
            if len(common) >= 3:
                return True
        return False
    
    def find_k33_violation(adj):
        """Find a specific K3,3 violation. Returns (A, B) where A,B are 3-sets, or None."""
        nbrs = [set() for _ in range(n)]
        for (u, v) in adj:
            nbrs[u].add(v)
            nbrs[v].add(u)
        for a, b, c in combinations(range(n), 3):
            common = nbrs[a] & nbrs[b] & nbrs[c]
            if len(common) >= 3:
                B = list(common)[:3]
                A = [a, b, c]
                return A, B
        return None
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v). Returns True if added and valid."""
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in adj:
            return True  # already there
        adj.add((u, v))
        if has_k33(adj):
            adj.remove((u, v))
            return False
        return True
    
    # Initial construction: start with a random graph, then repair
    random.seed(42)
    
    best_adj = set()
    best_count = 0
    
    # Try multiple random initializations
    for trial in range(50):
        if time.time() > deadline - 20:
            break
        
        # Random initial graph with edge probability ~0.4
        adj = set()
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.4:
                    adj.add((u, v))
        
        # Repair phase: remove edges until no K3,3
        for repair_iter in range(1000):
            if time.time() > deadline - 10:
                break
            viol = find_k33_violation(adj)
            if viol is None:
                break
            A, B = viol
            # Remove a random edge from the K3,3
            edges_in_k33 = [(a, b) for a in A for b in B]
            # Normalize
            edges_in_k33 = [tuple(sorted(e)) for e in edges_in_k33]
            edges_in_k33 = [e for e in edges_in_k33 if e in adj]
            if edges_in_k33:
                e_to_remove = random.choice(edges_in_k33)
                adj.remove(e_to_remove)
            else:
                break
        
        # Greedy addition phase
        for add_pass in range(10):
            if time.time() > deadline - 10:
                break
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) not in adj:
                        if try_add_edge(adj, u, v):
                            improved = True
            if not improved:
                break
        
        # Local search: edge swaps
        for swap_pass in range(200):
            if time.time() > deadline - 10:
                break
            improved = False
            # Try removing one edge and adding another
            edges_list = list(adj)
            random.shuffle(edges_list)
            for e_remove in edges_list:
                if time.time() > deadline - 10:
                    break
                adj.remove(e_remove)
                # Try to add edges
                for u in range(n):
                    for v in range(u+1, n):
                        if (u, v) not in adj and (u, v) != e_remove:
                            if try_add_edge(adj, u, v):
                                improved = True
                                break
                    if improved:
                        break
                # If we didn't improve, put the edge back
                if not improved:
                    adj.add(e_remove)
                    break  # move to next edge to remove
                else:
                    break
            
            if not improved:
                break
        
        # Update best
        if len(adj) > best_count:
            best_count = len(adj)
            best_adj = set(adj)
    
    # Final greedy addition on best
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in best_adj:
                if try_add_edge(best_adj, u, v):
                    pass
    
    # Convert to edge list
    result = [[u, v] for (u, v) in best_adj]
    return result
