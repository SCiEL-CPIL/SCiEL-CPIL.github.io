import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    mandatory = [1, 2, 5, 11]
    N = 284
    
    # 1. Identify candidates compatible with mandatory
    # Sums of mandatory
    mand_sums = set()
    for i in range(len(mandatory)):
        for j in range(i, len(mandatory)):
            mand_sums.add(mandatory[i] + mandatory[j])
            
    candidates = []
    for x in range(12, N + 1):
        # Check if x is compatible with mandatory
        compatible = True
        for m in mandatory:
            s = x + m
            if s in mand_sums:
                compatible = False
                break
            # Also check if s collides with other x+m'? 
            # Wait, the condition is that ALL pairwise sums in M U {x} are distinct.
            # The sums in M are distinct.
            # The new sums are x+m for m in M, and 2x.
            # We need:
            # 1. 2x not in mand_sums
            # 2. x+m not in mand_sums
            # 3. x+m1 != x+m2 (always true if m1!=m2)
            # 4. x+m != 2x => m != x (true since x >= 12, m <= 11)
            # So we just need 2x not in mand_sums and x+m not in mand_sums.
        if 2*x in mand_sums:
            continue
        # Check x+m
        ok = True
        for m in mandatory:
            if x + m in mand_sums:
                ok = False
                break
        if ok:
            candidates.append(x)
            
    # 2. Build conflict graph among candidates
    # Two candidates u, v conflict if M U {u, v} is not Sidon.
    # This happens if:
    # a) u+v is in mand_sums
    # b) u+m == v+m' (implies u-v = m'-m). Since m, m' in M, u-v is fixed diff.
    #    Actually, we just need to check if any sum in {u+v, 2u, 2v, u+m, v+m} collides with existing or each other.
    #    Existing sums: mand_sums.
    #    New sums: 2u, 2v, u+v, u+m, v+m.
    #    We know 2u, u+m are not in mand_sums (from filter).
    #    We know 2v, v+m are not in mand_sums (from filter).
    #    So we only need to check collisions between the new sums themselves, and with mand_sums.
    #    Collisions with mand_sums:
    #    - u+v in mand_sums
    #    - u+m in mand_sums (already checked for u individually? No, u+m was checked against mand_sums. 
    #      Wait, in step 1, I checked if x+m in mand_sums. So u+m is NOT in mand_sums. 
    #      So u+m cannot collide with mand_sums.
    #      Similarly v+m is not in mand_sums.
    #      So only u+v can collide with mand_sums.
    #    Collisions between new sums:
    #    - 2u == 2v (u!=v, false)
    #    - 2u == u+v => u==v (false)
    #    - 2u == v+m => 2u - v = m. m in M.
    #    - 2v == u+m => 2v - u = m. m in M.
    #    - u+m1 == v+m2 => u-v = m2-m1.
    #    - u+v == 2u (false)
    #    - u+v == 2v (false)
    #    - u+v == u+m => v=m (false, v>=12)
    #    - u+v == v+m => u=m (false)
    
    # So conflicts are:
    # 1. u + v in mand_sums
    # 2. 2u - v in mandatory
    # 3. 2v - u in mandatory
    # 4. u - v in {m1 - m2 | m1, m2 in mandatory, m1 != m2}
    
    diffs = set()
    for i in range(len(mandatory)):
        for j in range(len(mandatory)):
            if i != j:
                diffs.add(mandatory[i] - mandatory[j])
                
    n_cand = len(candidates)
    # Map candidate value to index
    cand_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Adjacency list
    adj = [set() for _ in range(n_cand)]
    
    for i in range(n_cand):
        u = candidates[i]
        for j in range(i + 1, n_cand):
            v = candidates[j]
            conflict = False
            if u + v in mand_sums:
                conflict = True
            elif 2*u - v in mandatory:
                conflict = True
            elif 2*v - u in mandatory:
                conflict = True
            elif (u - v) in diffs:
                conflict = True
            if conflict:
                adj[i].add(j)
                adj[j].add(i)
                
    # 3. Solve MIS on this graph
    # Use Tomita's algorithm
    best_set = []
    
    # Color the graph for upper bounds
    # We need a greedy coloring function
    def greedy_color(vertices, adj):
        # Returns list of colors for vertices in the given list
        # vertices is a list of indices
        colors = {}
        color_map = []
        for v in vertices:
            # Find smallest color not used by neighbors in vertices
            used = set()
            for u in adj[v]:
                if u in colors:
                    used.add(colors[u])
            c = 1
            while c in used:
                c += 1
            colors[v] = c
        return colors
        
    # Tomita's expand
    def expand(R, P):
        nonlocal best_set
        if not P:
            if len(R) > len(best_set):
                best_set = R[:]
            return
            
        # Color P
        colors = greedy_color(P, adj)
        # Sort P by color
        P_sorted = sorted(P, key=lambda x: colors[x])
        
        # Iterate in reverse
        for i in range(len(P_sorted) - 1, -1, -1):
            v = P_sorted[i]
            # Upper bound: |R| + colors[v]
            if len(R) + colors[v] <= len(best_set):
                break
                
            # Branch 1: Include v
            R.append(v)
            # New P is neighbors of v in P
            P_new = [u for u in P_sorted[:i+1] if u in adj[v]]
            expand(R, P_new)
            R.pop()
            
            # Branch 2: Exclude v
            # Remove v from P
            P_sorted[i] = P_sorted[-1]
            P_sorted.pop()
            
    # Initialize
    all_vertices = list(range(n_cand))
    expand([], all_vertices)
    
    # Convert best_set indices back to values
    best_vals = [candidates[i] for i in best_set]
    
    # Combine with mandatory
    result = mandatory + best_vals
    
    # Verify
    def is_valid(s):
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sm = s[i] + s[j]
                if sm in sums:
                    return False
                sums.add(sm)
        return True
        
    if not is_valid(result):
        # Fallback to greedy if something went wrong
        a = mandatory[:]
        sums = set()
        for i, x in enumerate(a):
            for y in a[i:]:
                sums.add(x + y)
        for x in range(12, 285):
            new = {x + y for y in a} | {2 * x}
            if len(new) == len(a) + 1 and not new & sums:
                a.append(x)
                sums |= new
        return a
        
    return result
