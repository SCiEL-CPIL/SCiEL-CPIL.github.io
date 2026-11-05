import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 40s safety margin for evaluator overhead
    
    def clebsch_complement():
        """Return the complement of the Clebsch graph on 16 vertices.
        Clebsch graph: vertices are binary strings of length 4.
        Edge if Hamming distance is 1 or 3.
        Complement: edge if Hamming distance is 2.
        This is a (5,5)-free graph."""
        n = 16
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                # Hamming distance
                xor = i ^ j
                dist = bin(xor).count('1')
                if dist == 2:
                    edges.append((i, j))
        return n, edges
    
    def paley_graph(n):
        """Generate Paley graph on n vertices (n must be prime, n ≡ 1 mod 4)."""
        if n < 2:
            return 0, []
        def is_prime(x):
            if x < 2:
                return False
            if x == 2:
                return True
            if x % 2 == 0:
                return False
            for i in range(3, int(x**0.5) + 1, 2):
                if x % i == 0:
                    return False
            return True
        if not is_prime(n) or n % 4 != 1:
            return 0, []
        
        residues = set()
        for x in range(1, n):
            residues.add((x * x) % n)
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n in residues:
                    edges.append((i, j))
        return n, edges
    
    def build_bitmasks(n, edges):
        """Build bitmask adjacency representation."""
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        return adj_masks
    
    def count_k5_and_ind5(n, adj_masks):
        """Count K5s and independent 5-sets using bitmasks. Returns (k5_count, ind5_count)."""
        k5_count = 0
        ind5_count = 0
        
        for combo in combinations(range(n), 5):
            a, b, c, d, e = combo
            # Check K5: all pairs must be edges
            if ((adj_masks[a] >> b) & 1) and ((adj_masks[a] >> c) & 1) and \
               ((adj_masks[a] >> d) & 1) and ((adj_masks[a] >> e) & 1) and \
               ((adj_masks[b] >> c) & 1) and ((adj_masks[b] >> d) & 1) and \
               ((adj_masks[b] >> e) & 1) and ((adj_masks[c] >> d) & 1) and \
               ((adj_masks[c] >> e) & 1) and ((adj_masks[d] >> e) & 1):
                k5_count += 1
            # Check independent 5-set: no pairs must be edges
            elif not ((adj_masks[a] >> b) & 1) and not ((adj_masks[a] >> c) & 1) and \
                 not ((adj_masks[a] >> d) & 1) and not ((adj_masks[a] >> e) & 1) and \
                 not ((adj_masks[b] >> c) & 1) and not ((adj_masks[b] >> d) & 1) and \
                 not ((adj_masks[b] >> e) & 1) and not ((adj_masks[c] >> d) & 1) and \
                 not ((adj_masks[c] >> e) & 1) and not ((adj_masks[d] >> e) & 1):
                ind5_count += 1
        
        return k5_count, ind5_count
    
    def find_one_violation(n, adj_masks):
        """Find one K5 or independent 5-set. Returns (type, vertices) or None."""
        for combo in combinations(range(n), 5):
            a, b, c, d, e = combo
            # Check K5
            if ((adj_masks[a] >> b) & 1) and ((adj_masks[a] >> c) & 1) and \
               ((adj_masks[a] >> d) & 1) and ((adj_masks[a] >> e) & 1) and \
               ((adj_masks[b] >> c) & 1) and ((adj_masks[b] >> d) & 1) and \
               ((adj_masks[b] >> e) & 1) and ((adj_masks[c] >> d) & 1) and \
               ((adj_masks[c] >> e) & 1) and ((adj_masks[d] >> e) & 1):
                return ("clique", combo)
            # Check independent
            if not ((adj_masks[a] >> b) & 1) and not ((adj_masks[a] >> c) & 1) and \
                 not ((adj_masks[a] >> d) & 1) and not ((adj_masks[a] >> e) & 1) and \
                 not ((adj_masks[b] >> c) & 1) and not ((adj_masks[b] >> d) & 1) and \
                 not ((adj_masks[b] >> e) & 1) and not ((adj_masks[c] >> d) & 1) and \
                 not ((adj_masks[c] >> e) & 1) and not ((adj_masks[d] >> e) & 1):
                return ("independent", combo)
        return None
    
    def edges_from_bitmasks(n, adj_masks):
        """Convert bitmask adjacency to edge list."""
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append((i, j))
        return edges
    
    # Get seed graphs
    best_n = 0
    best_edges = []
    
    # Try Clebsch complement first
    cn, ce = clebsch_complement()
    k5c, ind5c = count_k5_and_ind5(cn, build_bitmasks(cn, ce))
    if k5c == 0 and ind5c == 0:
        best_n = cn
        best_edges = ce
    
    # Try Paley 37
    pn, pe = paley_graph(37)
    if pn > 0:
        adj_p = build_bitmasks(pn, pe)
        k5p, ind5p = count_k5_and_ind5(pn, adj_p)
        if k5p == 0 and ind5p == 0:
            if pn > best_n:
                best_n = pn
                best_edges = pe
    
    # Try other Paley graphs
    for q in [41, 43]:
        if q > 46:
            break
        qn, qe = paley_graph(q)
        if qn > 0:
            adj_q = build_bitmasks(qn, qe)
            k5q, ind5q = count_k5_and_ind5(qn, adj_q)
            if k5q == 0 and ind5q == 0:
                if qn > best_n:
                    best_n = qn
                    best_edges = qe
    
    if best_n == 0:
        # Fallback: Paley 37
        best_edges = paley_graph(37)[1]
        best_n = 37
    
    # Now try to expand from the best seed
    current_n = best_n
    current_adj = build_bitmasks(current_n, best_edges)
    
    # Try vertex duplication and expansion
    for target_n in range(current_n + 1, 47):
        if time.time() > deadline - 5:
            break
        
        found = False
        attempts = 0
        max_attempts = max(100, int((deadline - time.time()) / 0.5))
        
        while attempts < max_attempts and not found:
            if time.time() > deadline - 5:
                break
            attempts += 1
            
            # Start from current graph, add one vertex
            new_n = current_n + 1
            new_adj = [m for m in current_adj] + [0]
            
            # Assign edges to new vertex based on heuristic
            # Connect to vertices that would help avoid K5 and ind5
            for i in range(current_n):
                # Compute how many K5s and ind5s this vertex would be in
                # if connected/not connected to i
                # Simple heuristic: random with bias toward avoiding violations
                if random.random() < 0.5:
                    new_adj[i] |= (1 << new_n)
                    new_adj[new_n] |= (1 << i)
            
            # Check for violations
            violation = find_one_violation(new_n, new_adj)
            if violation is None:
                # Valid! Accept
                current_n = new_n
                current_adj = new_adj
                found = True
                break
            else:
                # Try to fix by flipping some edges
                vtype, vset = violation
                # Try flipping edges within the violation set
                pairs = list(combinations(vset, 2))
                random.shuffle(pairs)
                
                fixed = False
                for u, v in pairs:
                    if u > v:
                        u, v = v, u
                    # Flip edge (u, v)
                    if (new_adj[u] >> v) & 1:
                        new_adj[u] &= ~(1 << v)
                        new_adj[v] &= ~(1 << u)
                    else:
                        new_adj[u] |= (1 << v)
                        new_adj[v] |= (1 << u)
                    
                    # Check if violation is fixed
                    violation2 = find_one_violation(new_n, new_adj)
                    if violation2 is None:
                        current_n = new_n
                        current_adj = new_adj
                        found = True
                        fixed = True
                        break
                    else:
                        # Revert
                        if (new_adj[u] >> v) & 1:
                            new_adj[u] &= ~(1 << v)
                            new_adj[v] &= ~(1 << u)
                        else:
                            new_adj[u] |= (1 << v)
                            new_adj[v] |= (1 << u)
                
                if fixed:
                    break
        
        if not found:
            # Try local search on current_n to see if we can improve
            # or if we should try a different approach
            if time.time() > deadline - 10:
                break
            
            # Try simulated annealing on current graph
            improved = False
            for sa_iter in range(min(500, int((deadline - time.time()) / 0.02))):
                if time.time() > deadline - 5:
                    break
                
                # Pick random edge to flip
                u = random.randint(0, current_n - 1)
                v = random.randint(0, current_n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                # Count violations before
                old_violation = find_one_violation(current_n, current_adj)
                
                # Flip
                if (current_adj[u] >> v) & 1:
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                
                new_violation = find_one_violation(current_n, current_adj)
                
                # Accept if no violation, or with some probability if still has violation
                if new_violation is None:
                    improved = True
                    break
                else:
                    # Revert with high probability
                    if random.random() < 0.7:
                        if (current_adj[u] >> v) & 1:
                            current_adj[u] &= ~(1 << v)
                            current_adj[v] &= ~(1 << u)
                        else:
                            current_adj[u] |= (1 << v)
                            current_adj[v] |= (1 << u)
            
            if improved:
                # Try to add another vertex
                continue
            else:
                break
    
    # Final verification
    final_adj = build_bitmasks(current_n, edges_from_bitmasks(current_n, current_adj))
    k5f, ind5f = count_k5_and_ind5(current_n, final_adj)
    
    if k5f == 0 and ind5f == 0:
        final_edges = edges_from_bitmasks(current_n, final_adj)
        return current_n, final_edges
    else:
        # Return the last known valid graph
        return best_n, best_edges

