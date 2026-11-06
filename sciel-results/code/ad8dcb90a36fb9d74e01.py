import itertools
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s margin
    
    # Try n from 17 down to 1
    for n in range(17, 0, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Precompute C6s for this n
        # A C6 is a tuple of 6 vertices (v0, ..., v5) such that they form a cycle.
        # To avoid duplicates, we fix v0 < v1 < ... ? No, cycle order matters for edge list.
        # But for "monochromatic C6", we just need the set of 6 edges.
        # A C6 is defined by the set of 6 edges.
        # Two different vertex orderings might give the same set of edges (reversal).
        # We just need to ensure we check each unique set of 6 edges once.
        # Number of unique C6s in K_n is C(n,6) * 60.
        
        c6s = []
        # Map edge (i,j) to list of c6 indices
        # Edges are indexed by (i,j) with i<j.
        # We need a fast way to get edge index.
        # edge_idx = i * (n-1) - i*(i-1)//2 + (j-i-1) ?
        # Simpler: use a dict or a 2D array of size n x n.
        
        # Let's generate all C6s
        # To generate unique C6s (up to rotation and reversal), we can fix v0=0?
        # No, we need ALL C6s in the graph.
        # Just generate all permutations of 6-subsets and deduplicate by edge set?
        # That's expensive.
        
        # Better:
        # For each 6-subset S, there are 60 distinct cycles.
        # We can just generate the 60 cycles for each subset.
        # Total C6s = C(n,6) * 60.
        
        all_c6s = []
        # Precompute edge indices for fast lookup
        # edge_id[i][j]
        edge_id = [[-1]*n for _ in range(n)]
        edge_list = []
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_id[i][j] = idx
                edge_id[j][i] = idx
                edge_list.append((i,j))
                idx += 1
        
        # Generate C6s
        # For each combination of 6 vertices
        for subset in itertools.combinations(range(n), 6):
            # Generate 60 unique cycles
            # Fix v0 = subset[0]
            # Permute the other 5
            # To avoid reversal duplicates, we can impose an ordering?
            # Actually, for the purpose of checking "monochromatic", 
            # a cycle and its reverse are the same set of edges.
            # So we only need 30 cycles per subset?
            # No, the set of edges is the same.
            # So we can just generate all 60 and deduplicate by frozenset of edges?
            # Or just generate 30 by fixing v0 and requiring v1 < v5?
            
            v0 = subset[0]
            rest = list(subset[1:])
            seen_edge_sets = set()
            for perm in itertools.permutations(rest):
                v = [v0] + list(perm)
                # Check if this is a new cycle (edge set)
                # Edges: (v0,v1), (v1,v2), ..., (v5,v0)
                edges = []
                for k in range(6):
                    u, w = v[k], v[(k+1)%6]
                    if u > w: u, w = w, u
                    edges.append((u,w))
                es = frozenset(edges)
                if es not in seen_edge_sets:
                    seen_edge_sets.add(es)
                    all_c6s.append(es)
                    
        # all_c6s is a list of frozensets of edges.
        # We need to map edge -> list of c6 indices.
        c6s_for_edge = defaultdict(list)
        for idx_c6, es in enumerate(all_c6s):
            for e in es:
                c6s_for_edge[e].append(idx_c6)
                
        # Initial random coloring
        num_edges = len(edge_list)
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Initial cost and counts
        # c6_counts[c6_idx][color]
        c6_counts = [[0]*4 for _ in range(len(all_c6s))]
        cost = 0
        
        for idx_c6, es in enumerate(all_c6s):
            # Count colors in this C6
            for e in es:
                # Get edge index
                # e is (u,v)
                e_idx = edge_id[e[0]][e[1]]
                c = colors[e_idx]
                c6_counts[idx_c6][c] += 1
            # Check if any color has count 6
            for c in range(4):
                if c6_counts[idx_c6][c] == 6:
                    cost += 1
                    
        if cost == 0:
            return {'n': n, 'colors': colors}
            
        # WalkSAT
        # We need to pick edges to flip.
        # We will iterate until time runs out or cost is 0.
        
        # To speed up, we can prioritize edges in violated C6s.
        # But simple random is okay for now.
        
        while time.time() - start_time < time_limit:
            # Pick a random edge
            e_idx = random.randint(0, num_edges - 1)
            e = edge_list[e_idx]
            
            old_c = colors[e_idx]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
                
            # Calculate delta cost
            delta = 0
            affected_c6s = c6s_for_edge[e]
            
            for idx_c6 in affected_c6s:
                # Update counts temporarily
                # c6_counts[idx_c6][old_c] -= 1
                # c6_counts[idx_c6][new_c] += 1
                
                old_count_old = c6_counts[idx_c6][old_c]
                old_count_new = c6_counts[idx_c6][new_c]
                
                new_count_old = old_count_old - 1
                new_count_new = old_count_new + 1
                
                # Check violations
                old_viol = 1 if old_count_old == 6 else 0
                # Note: if old_count_old was 6, it was a violation.
                # If old_count_new was 6, it was a violation? No, only one color can be 6.
                # Actually, sum of counts is 6. So only one color can be 6.
                
                new_viol = 1 if new_count_new == 6 else 0
                # Also, if old_count_old was 6, the old state had a violation.
                # If new_count_new is 6, the new state has a violation.
                
                # Wait, what if old_count_old was 6 AND new_count_new is 6? Impossible, counts sum to 6.
                
                delta += new_viol - old_viol
                
            # Acceptance probability
            if delta < 0:
                # Apply
                colors[e_idx] = new_c
                for idx_c6 in affected_c6s:
                    c6_counts[idx_c6][old_c] -= 1
                    c6_counts[idx_c6][new_c] += 1
                cost += delta
                if cost == 0:
                    return {'n': n, 'colors': colors}
            else:
                # Simulated Annealing or Random Accept
                # Simple: accept with prob 0.1
                if random.random() < 0.1:
                    colors[e_idx] = new_c
                    for idx_c6 in affected_c6s:
                        c6_counts[idx_c6][old_c] -= 1
                        c6_counts[idx_c6][new_c] += 1
                    cost += delta
                    if cost == 0:
                        return {'n': n, 'colors': colors}
                        
    # Fallback
    return {'n': 1, 'colors': []}
