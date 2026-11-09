def _backtrack(k, time_budget, seed):
    rng = random.Random(seed)
    start = time.time()
    deadline = start + time_budget
    colors = [-1]*N
    # symmetry: colors[0]=0
    colors[0] = 0
    # domains
    domains = [set(range(k)) for _ in range(N)]
    domains[0] = {0}
    
    # Precompute for each AP, which positions
    # Forward checking: after assignment, for each AP containing the position, check feasibility
    
    def feasible_ap(i, colors):
        a,b,c,d = APS[i]
        vals = [colors[a],colors[b],colors[c],colors[d]]
        assigned = [v for v in vals if v != -1]
        if len(assigned) == 4:
            return not _ap_viol_colors(*vals)
        # partial: check if satisfiable
        # distinct assigned
        s = set(assigned)
        unassigned_count = 4 - len(assigned)
        if len(s) >= 3:
            return True
        if len(s) == 2:
            # need unassigned to include a color not in s
            # (handled by domain restriction; here just check unassigned>0)
            return unassigned_count >= 1
        if len(s) == 1:
            # need 2 more distinct from unassigned
            return unassigned_count >= 2
        return True  # 0 assigned
    
    def propagate(p, c):
        # returns False if conflict
        colors[p] = c
        domains[p] = {c}
        for i in POS_APS[p]:
            a,b,c2,d = APS[i]
            if not feasible_ap(i, colors):
                return False
            # restrict domains of unassigned in this AP
            s = set(v for v in (colors[a],colors[b],colors[c2],colors[d]) if v != -1)
            unassigned = [q for q in (a,b,c2,d) if colors[q] == -1]
            if len(s) == 2 and unassigned:
                # each unassigned must be able to take a color not in s
                for q in unassigned:
                    domains[q] = {x for x in domains[q] if x not in s}
                    if not domains[q]:
                        return False
            elif len(s) == 1 and len(unassigned) == 1:
                q = unassigned[0]
                # need q to add 2 distinct? no, s has 1, need total>=3, unassigned=1 can add 1 -> total 2 <3. conflict!
                return False
            elif len(s) == 1 and len(unassigned) == 2:
                # the 2 unassigned must be 2 distinct colors not in s
                c0 = next(iter(s))
                for q in unassigned:
                    domains[q] = {x for x in domains[q] if x != c0}
                    if not domains[q]:
                        return False
        return True
    
    # MRV: pick unassigned with smallest domain
    def select_var():
        best = -1
        best_size = float('inf')
        for p in range(N):
            if colors[p] == -1:
                sz = len(domains[p])
                if sz < best_size:
                    best_size = sz
                    best = p
                    if sz <= 1:
                        break
        return best
    
    def solve():
        if time.time() > deadline:
            return 'timeout'
        p = select_var()
        if p == -1:
            return True
        vals = list(domains[p])
        rng.shuffle(vals)
        for c in vals:
            # save state
            save_colors = colors.copy()
            save_domains = [set(d) for d in domains]
            if propagate(p, c):
                r = solve()
                if r is True:
                    return True
                if r == 'timeout':
                    return 'timeout'
            # restore
            colors[:] = save_colors
            domains[:] = [set(d) for d in save_domains]
        return False
    
    r = solve()
    if r is True:
        return colors
    return None
