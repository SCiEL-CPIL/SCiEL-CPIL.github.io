import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj):
        n = len(adj)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection
            if P:
                pivot = max(P, key=lambda u: len(P & adj[u]))
                candidates = P - adj[pivot]
            else:
                candidates = P
                
            for v in list(candidates):
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.remove(v)
                P.remove(v)
                X.add(v)
                
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(adj):
        n = len(adj)
        cliques = bron_kerbosch(adj)
        if not cliques:
            return []
        
        num_cliques = len(cliques)
        
        # Build masks: mask[v] has bit i set if v is in clique i
        masks = [0] * n
        for i, c in enumerate(cliques):
            bit = 1 << i
            for v in c:
                masks[v] |= bit
        
        all_mask = (1 << num_cliques) - 1
        
        # Precompute: for each clique, which vertices cover it
        clique_to_vertices = [set() for _ in range(num_cliques)]
        for v in range(n):
            m = masks[v]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                clique_to_vertices[idx].add(v)
                m ^= lsb
        
        # Clique rarity: how many vertices cover this clique
        clique_rarity = [0] * num_cliques
        for i in range(num_cliques):
            clique_rarity[i] = len(clique_to_vertices[i])
        
        best_solution = None
        
        for restart in range(300):
            if time.time() > deadline - 3.0:
                break
            
            rng = random.Random(restart * 7919 + 12345)
            uncovered = all_mask
            selected = []
            available = set(range(n))
            
            # Greedy: select vertices that cover the most uncovered cliques, weighted by rarity
            while uncovered:
                best_v = -1
                max_score = -1
                for v in available:
                    m = masks[v] & uncovered
                    if not m:
                        continue
                    # Score: sum of 1/rarity for each uncovered clique covered
                    score = 0.0
                    mm = m
                    while mm:
                        lsb = mm & -mm
                        idx = lsb.bit_length() - 1
                        score += 1.0 / clique_rarity[idx]
                        mm ^= lsb
                    # Add small random perturbation
                    score += rng.random() * 0.01
                    if score > max_score:
                        max_score = score
                        best_v = v
                
                if best_v == -1:
                    break
                
                selected.append(best_v)
                available.discard(best_v)
                uncovered &= ~masks[best_v]
            
            # Minimization: remove redundant vertices
            cover_count = [0] * num_cliques
            for v in selected:
                m = masks[v]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_count[idx] += 1
                    m ^= lsb
            
            changed = True
            while changed:
                changed = False
                for v in list(selected):
                    m = masks[v]
                    critical = False
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_count[idx] == 1:
                            critical = True
                            break
                        m ^= lsb
                    if not critical:
                        selected.remove(v)
                        m = masks[v]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] -= 1
                            m ^= lsb
                        changed = True
            
            # Local search: 1-exchange to reduce size
            if time.time() < deadline - 2.0:
                sel_set = set(selected)
                unselected = [v for v in range(n) if v not in sel_set]
                
                # Recompute cover counts
                cover_count = [0] * num_cliques
                for v in selected:
                    m = masks[v]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] += 1
                        m ^= lsb
                
                improved = True
                while improved and time.time() < deadline - 2.0:
                    improved = False
                    for v_out in list(selected):
                        m_out = masks[v_out]
                        # Find cliques only covered by v_out
                        critical_cliques = []
                        mm = m_out
                        while mm:
                            lsb = mm & -mm
                            idx = lsb.bit_length() - 1
                            if cover_count[idx] == 1:
                                critical_cliques.append(idx)
                            mm ^= lsb
                        
                        if not critical_cliques:
                            continue
                        
                        # Find v_in that covers all critical cliques
                        for v_in in unselected:
                            m_in = masks[v_in]
                            valid = True
                            for idx in critical_cliques:
                                if not ((m_in >> idx) & 1):
                                    valid = False
                                    break
                            if not valid:
                                continue
                            
                            # Try swap and minimize
                            new_selected = [v for v in selected if v != v_out] + [v_in]
                            new_cover_count = [0] * num_cliques
                            for v in new_selected:
                                m = masks[v]
                                while m:
                                    lsb = m & -m
                                    idx = lsb.bit_length() - 1
                                    new_cover_count[idx] += 1
                                    m ^= lsb
                            
                            # Minimize new_selected
                            new_sel = list(new_selected)
                            new_changed = True
                            while new_changed:
                                new_changed = False
                                for v in list(new_sel):
                                    m = masks[v]
                                    crit = False
                                    while m:
                                        lsb = m & -m
                                        idx = lsb.bit_length() - 1
                                        if new_cover_count[idx] == 1:
                                            crit = True
                                            break
                                        m ^= lsb
                                    if not crit:
                                        new_sel.remove(v)
                                        m = masks[v]
                                        while m:
                                            lsb = m & -m
                                            idx = lsb.bit_length() - 1
                                            new_cover_count[idx] -= 1
                                            m ^= lsb
                                        new_changed = True
                            
                            if len(new_sel) < len(selected):
                                selected = new_sel
                                sel_set = set(selected)
                                unselected = [v for v in range(n) if v not in sel_set]
                                improved = True
                                # Recompute cover_count for next iteration
                                cover_count = [0] * num_cliques
                                for v in selected:
                                    m = masks[v]
                                    while m:
                                        lsb = m & -m
                                        idx = lsb.bit_length() - 1
                                        cover_count[idx] += 1
                                        m ^= lsb
                                break
                        if improved:
                            break
            
            if best_solution is None or len(selected) < len(best_solution):
                best_solution = list(selected)
            
            if len(best_solution) <= 3:
                break

        return sorted(best_solution) if best_solution else []

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        adj = random_graph(n, p, seed)
        sol = solve_instance(adj)
        out.append(sol)
        
    return out
