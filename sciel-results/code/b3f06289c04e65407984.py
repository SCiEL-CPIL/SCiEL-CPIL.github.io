    def solve_instance_v2(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Convert lists to bitmask domains
        # Colors are 0..4, so 5 bits needed.
        doms = [0] * n
        for i, l in enumerate(lists):
            mask = 0
            for c in l:
                mask |= (1 << c)
            doms[i] = mask
            
        if any(d == 0 for d in doms):
            return None

        # AC-3
        # Queue of arcs (i, j)
        queue = []
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        
        head = 0
        while head < len(queue):
            i, j = queue[head]
            head += 1
            # Revise i with respect to j
            # If doms[j] is singleton {c}, remove c from doms[i]
            dj = doms[j]
            if dj != 0 and (dj & (dj - 1)) == 0: # dj is power of 2
                c = dj.bit_length() - 1
                if doms[i] & (1 << c):
                    doms[i] &= ~(1 << c)
                    if doms[i] == 0:
                        return None
                    for k in adj[i]:
                        if k != j:
                            queue.append((k, i))

        if any(d == 0 for d in doms):
            return None

        # Backtracking with bitmasks
        import sys
        sys.setrecursionlimit(10000)
        
        def bt(doms, assign):
            if time.time() > DEADLINE:
                raise TimeoutError()
            
            # MRV: Find unassigned node with min domain size
            min_sz = float('inf')
            min_v = -1
            for i in range(n):
                if i not in assign:
                    sz = doms[i].bit_count()
                    if sz < min_sz:
                        min_sz = sz
                        min_v = i
                        if sz == 0:
                            return None
                        if sz == 1:
                            break
            if min_v == -1:
                return assign # Solution found

            # LVC: Sort values by least constraining value
            # Values are bits set in doms[min_v]
            vals = []
            d = doms[min_v]
            while d:
                lsb = d & -d
                c = lsb.bit_length() - 1
                vals.append(c)
                d ^= lsb
            
            # Impact: number of unassigned neighbors having this color
            def impact(c):
                cnt = 0
                mask_c = 1 << c
                for nb in adj[min_v]:
                    if nb not in assign:
                        if doms[nb] & mask_c:
                            cnt += 1
                return cnt
            vals.sort(key=impact)

            for c in vals:
                mask_c = 1 << c
                # Check validity against assigned neighbors
                valid = True
                for nb in adj[min_v]:
                    if nb in assign:
                        if assign[nb] == c:
                            valid = False
                            break
                if not valid:
                    continue
                
                # Propagate
                new_doms = doms[:]
                new_doms[min_v] = mask_c
                for nb in adj[min_v]:
                    if nb not in assign:
                        if new_doms[nb] & mask_c:
                            new_doms[nb] &= ~mask_c
                            if new_doms[nb] == 0:
                                valid = False
                                break
                if not valid:
                    continue
                
                new_assign = dict(assign)
                new_assign[min_v] = c
                
                res = bt(new_doms, new_assign)
                if res is not None:
                    return res
            return None

        try:
            sol_dict = bt(doms, {})
            if sol_dict:
                return [int(sol_dict[i]) for i in range(n)]
        except TimeoutError:
            pass
        return None
