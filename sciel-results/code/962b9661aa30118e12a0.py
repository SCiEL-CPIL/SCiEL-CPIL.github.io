import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 512
    # Build adjacency: node i (0-indexed) represents value i+1
    adj = [0] * n
    
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            l = a * b // gcd(a, b)
            if l <= 512:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Compute degrees
    degrees = [adj[i].bit_count() for i in range(n)]
    
    # Order nodes by degree (descending) for better search
    order = sorted(range(n), key=lambda x: -degrees[x])
    # Remap: new_index -> old_index
    # Actually, let's keep original indices but use degree ordering in pivot selection
    
    best_clique = []
    best_size = 0
    
    # Initial good clique: divisors of 360
    divs_360 = [d for d in range(1, 361) if 360 % d == 0]
    best_clique = divs_360[:]
    best_size = len(divs_360)
    
    # Greedy coloring for upper bound (Tomita style)
    # color[v] = color class number for vertex v in the ordering
    # We'll use a greedy coloring on the subgraph induced by P
    
    def greedy_color(P, order_list):
        """Assign colors to nodes in P using greedy coloring.
        Returns (colors_dict, max_color) where colors_dict maps node->color.
        """
        colors = {}
        color_classes = []  # list of sets, color_classes[c] = set of nodes with color c
        for v in order_list:
            if not (P >> v) & 1:
                continue
            # Find smallest color not used by neighbors of v in P
            used = set()
            for u in range(n):
                if (P >> u) & 1 and (adj[v] >> u) & 1:
                    if u in colors:
                        used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            if c >= len(color_classes):
                color_classes.append(set())
            color_classes[c].add(v)
        max_color = max(colors.values()) + 1 if colors else 0
        return colors, max_color
    
    def bronk(R, P, X):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return
        if not P and not X:
            sz = R.bit_count()
            if sz > best_size:
                best_size = sz
                best_clique = [i + 1 for i in range(n) if (R >> i) & 1]
            return
        if not P:
            return
        
        # Upper bound: |R| + |P|
        if R.bit_count() + P.bit_count() <= best_size:
            return
        
        # Get list of nodes in P
        P_nodes = []
        temp = P
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            P_nodes.append(v)
            temp ^= lsb
        
        if not P_nodes:
            return
        
        # Greedy coloring for upper bound
        colors, max_color = greedy_color(P, P_nodes)
        
        # Upper bound: |R| + max_color
        if R.bit_count() + max_color <= best_size:
            return
        
        # Choose pivot: node in P∪X maximizing |P ∩ N(u)|
        union = P | X
        pivot = -1
        max_count = -1
        temp_u = union
        while temp_u:
            lsb = temp_u & (-temp_u)
            u = lsb.bit_length() - 1
            temp_u ^= lsb
            count = (P & adj[u]).bit_count()
            if count > max_count:
                max_count = count
                pivot = u
                if count == P.bit_count():
                    break
        
        if pivot == -1:
            pivot = P_nodes[0]
        
        # Iterate over P \ N(pivot), ordered by color (descending)
        candidates = P & ~adj[pivot]
        # Get candidate nodes sorted by color descending
        cand_nodes = []
        temp_c = candidates
        while temp_c:
            lsb = temp_c & (-temp_c)
            v = lsb.bit_length() - 1
            cand_nodes.append(v)
            temp_c ^= lsb
        
        # Sort by color descending for better pruning
        cand_nodes.sort(key=lambda v: -colors.get(v, 0))
        
        for v in cand_nodes:
            if time.time() > deadline:
                return
            lsb = 1 << v
            if not (P & lsb):
                continue
            
            # Upper bound check with coloring
            # The subgraph P ∩ N(v) has at most max_color - (number of colors > colors[v] that are used)
            # Simpler: just recurse
            bronk(R | lsb, P & adj[v], X & adj[v])
            
            P ^= lsb
            X |= lsb
            
            # After removing v, the coloring of remaining P nodes is still valid
            # but max_color might decrease; we don't recompute for efficiency
    
    all_nodes = (1 << n) - 1
    bronk(0, all_nodes, 0)
    
    if not best_clique:
        return [1]
    
    return best_clique
