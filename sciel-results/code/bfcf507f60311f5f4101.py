    # Track top solutions
    top_solutions = []
    # ... inside loop ...
    top_solutions.append((len(sel), sel))
    top_solutions.sort(key=lambda x: x[0])
    if len(top_solutions) > 10:
        top_solutions.pop()

    # After loop, build pool from top_solutions
    candidate_pool = set()
    for size, sel in top_solutions:
        for idx in sel:
            candidate_pool.add(idx)
