import random
import time

def construct():
    """
    Construct a K3,3-free graph on 20 vertices maximizing edges.
    """
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid_addition(adj, u, v):
        """
        Check if adding edge (u, v) to the current graph (represented by adj)
        creates a K3,3 subgraph.
        """
        # We need to check if there exist a1, a2, b1, b2 such that:
        # A = {u, a1, a2}, B = {v, b1, b2}
        # u connected to v, b1, b2
        # a1 connected to v, b1, b2
        # a2 connected to v, b1, b2
        
        # This implies:
        # b1, b2 must be in N(u)
        # a1, a2 must be in N(v) AND N(b1) AND N(b2)
        
        nu = adj[u]
        nv = adj[v]
        
        # We only consider b1, b2 from N(u) that are NOT v (since u-v is being added, v is not in N(u) yet in the check context? 
        # Actually, adj is the state BEFORE adding u-v. So v is not in nu.
        # But b1, b2 must be connected to u. So b1, b2 in nu.
        
        # Optimization: 
        # If |N(u)| < 2 or |N(v)| < 2, no K3,3 can be formed involving u-v as the "new" edge connecting the partitions?
        # Actually, if |N(u)| < 2, we can't pick b1, b2.
        # If |N(v)| < 2, we can't pick a1, a2 easily? 
        # a1, a2 must be in N(v). So if |N(v)| < 2, impossible.
        
        if len(nu) < 2 or len(nv) < 2:
            return True
            
        # Iterate over pairs in N(u)
        nu_list = list(nu)
        for i in range(len(nu_list)):
            b1 = nu_list[i]
            for j in range(i + 1, len(nu_list)):
                b2 = nu_list[j]
                
                # We need a1, a2 in N(v) ∩ N(b1) ∩ N(b2)
                # Note: a1, a2 must be distinct from u, v, b1, b2.
                # Since graph is simple, N(x) doesn't contain x.
                # b1 is in N(u). Is b1 in N(b1)? No.
                # Is b1 in N(v)? Maybe. If b1 in N(v), then b1 is a candidate for a?
                # If a = b1, then b1 is in A and B? No, A and B are disjoint in bipartite definition.
                # So we must exclude b1, b2, u, v from the candidate set for a.
                
                common = nv & adj[b1] & adj[b2]
                # Remove u, v, b1, b2 from common
                common.discard(u)
                common.discard(v)
                common.discard(b1)
                common.discard(b2)
                
                if len(common) >= 2:
                    return False
                    
        return True

    def count_edges(adj):
        return sum(len(neigh) for neigh in adj) // 2

    best_edges = []
    best_count = 0
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Run multiple random restarts
    while time.time() < deadline:
        random.shuffle(all_edges)
        adj = [set() for _ in range(n)]
        
        for u, v in all_edges:
            if is_valid_addition(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
        
        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            # Store the edges
            best_edges = [[u, v] for u in range(n) for v in adj[u] if u < v]
            
        # If we reached a very high number, we might stop early?
        # ex(20, K3,3) is likely < 120. 
        # If best_count > 110, we are probably close to optimal.
        # But let's just keep running to be safe, or break if time is tight.
        
    return best_edges

if __name__ == "__main__":
    # Quick self-test
    edges = construct()
    print(f"Found graph with {len(edges)} edges")
