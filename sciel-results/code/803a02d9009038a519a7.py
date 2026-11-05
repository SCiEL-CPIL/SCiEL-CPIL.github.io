def perturb(rows, rng):
    rows=[list(r) for r in rows]
    n_pert = rng.randint(1,2)
    for _ in range(n_pert):
        if rng.random()<0.5:
            i=min(range(n_rows), key=lambda j:max(rows[j]))  # smallest row
        else:
            i=rng.randrange(n_rows)
        fb=forbidden_of(rows,i)
        rows[i]=greedy_row_rand(fb,rng)
    return rows
