<<<<<<< SEARCH
    # 3. Local search with add/remove/swap moves (inspiration 1 style, more aggressive)
    def try_add_fast(s, v):
        return len(conflict[v] & s) == 0

    def local_search_improve(s, time_budget):
        s = set(s)
        improved_flag = True
        while improved_flag and not time_up(time_budget):
            improved_flag = False

            # 1. Try to add any single node
            for v in candidates:
                if v not in s and try_add_fast(s, v):
                    s.add(v)
                    improved_flag = True

            # 2. Try to remove one node to add >= 1 node (prefer >= 2 for size gain)
            removal_improved = False
            for u in list(s):
                if time_up(time_budget):
                    break
                s.remove(u)
                
                addable = []
                for v in candidates:
                    if v not in s and v != u:
                        if try_add_fast(s, v):
                            addable.append(v)
                
                if len(addable) >= 2:
                    for v in addable:
                        s.add(v)
                    removal_improved = True
                    improved_flag = True
                    break
                elif len(addable) == 1:
                    # Swap: remove u, add v. Size same.
                    # Accept if new node has lower degree (potential for future growth)
                    deg_u = len(conflict[u] & s)
                    deg_v = len(conflict[addable[0]] & s)
                    if deg_v < deg_u:
                        s.add(addable[0])
                        removal_improved = True
                        improved_flag = True
                        break
                    else:
                        s.add(u)
                else:
                    s.add(u)

        return s
=======
    # 3. Local search with add/remove/swap moves (inspiration 1 style, more aggressive)
    def try_add_fast(s, v):
        return len(conflict[v] & s) == 0

    def local_search_improve(s, time_budget):
        s = set(s)
        improved_flag = True
        while improved_flag and not time_up(time_budget):
            improved_flag = False

            # 1. Try to add any single node
            for v in candidates:
                if v not in s and try_add_fast(s, v):
                    s.add(v)
                    improved_flag = True

            # 2. Try to remove one node to add >= 1 node (prefer >= 2 for size gain)
            removal_improved = False
            for u in list(s):
                if time_up(time_budget):
                    break
                s.remove(u)
                
                addable = []
                for v in candidates:
                    if v not in s and v != u:
                        if try_add_fast(s, v):
                            addable.append(v)
                
                if len(addable) >= 2:
                    for v in addable:
                        s.add(v)
                    removal_improved = True
                    improved_flag = True
                    break
                elif len(addable) == 1:
                    # Swap: remove u, add v. Size same.
                    # Accept if new node has lower degree (potential for future growth)
                    deg_u = len(conflict[u] & s)
                    deg_v = len(conflict[addable[0]] & s)
                    if deg_v < deg_u:
                        s.add(addable[0])
                        removal_improved = True
                        improved_flag = True
                        break
                    else:
                        s.add(u)
                else:
                    s.add(u)

            # 3. 2-opt swap: remove two nodes, add two or more new nodes
            if not improved_flag and not time_up(time_budget):
                s_list = list(s)
                n_s = len(s_list)
                for i in range(n_s):
                    if time_up(time_budget):
                        break
                    for j in range(i + 1, n_s):
                        if time_up(time_budget):
                            break
                        u1, u2 = s_list[i], s_list[j]
                        s.discard(u1)
                        s.discard(u2)
                        addable = []
                        for v in candidates:
                            if v not in s and v != u1 and v != u2:
                                if try_add_fast(s, v):
                                    addable.append(v)
                        if len(addable) >= 2:
                            for v in addable[:2]:
                                s.add(v)
                            improved_flag = True
                            break
                        else:
                            s.add(u1)
                            s.add(u2)
                    if improved_flag:
                        break

        return s
>>>>>>> REPLACE

