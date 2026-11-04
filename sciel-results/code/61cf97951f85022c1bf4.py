<<<<<<< SEARCH
    for q in [13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 109, 113, 137, 149, 157, 173]:
        if remaining() < 12:
            break
        res = build_ramanujan(q)
        if res is None:
            continue
        n, edges = res
        if n >= best_n:
            continue
        gc = check_girth(n, edges, g, deadline - 8)
        if gc == -1:
            continue
        if gc >= g:
            best, best_n = (n, edges), n
=======
    for q in [13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 109, 113, 137, 149, 157, 173]:
        if remaining() < 12:
            break
        res = build_ramanujan(q)
        if res is None:
            continue
        n, edges = res
        if n >= best_n:
            continue
        gc = check_girth_nb(n, edges, g, deadline - 8)
        if gc == -1:
            continue
        if gc >= g:
            best, best_n = (n, edges), n
>>>>>>> REPLACE
