    remaining = [True]*n
    cur_deg = [len(adj[i]) for i in range(n)]
    heap = [(cur_deg[i], i) for i in range(n)]
    heapq.heapify(heap)
    solution = []
    while heap:
        if time.time() > DEADLINE:
            break
        d, i = heapq.heappop(heap)
        if not remaining[i] or d != cur_deg[i]:
            continue
        solution.append(i)
        remaining[i] = False
        removed = [i]
        for j in adj[i]:
            if remaining[j]:
                remaining[j] = False
                removed.append(j)
        # update degrees of remaining vertices adjacent to any removed vertex
        for r in removed:
            for w in adj[r]:
                if remaining[w]:
                    cur_deg[w] -= 1
                    heapq.heappush(heap, (cur_deg[w], w))
