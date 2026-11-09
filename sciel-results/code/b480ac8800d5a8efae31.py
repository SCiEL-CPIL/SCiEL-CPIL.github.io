import time
import random

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s kill

    def norm_edge(u, v):
        if u > v:
            u, v = v, u
        return (u, v)

    # Build the 40-cycle rim (40 edges)
    edges_set = set()
    for i in range(N):
        edges_set.add(norm_edge(i, (i + 1) % N))

    # Add 19 chords from hub vertex 0 to vertices 2..20.
    # Chord (0,d) covers cycle lengths d+1 and 41-d.
    # d=2..20 covers all lengths 3..39 exactly (d=20 gives 21 twice).
    # Total: 40 + 19 = 59 edges. Objective = 19. This is the theoretical
    # minimum since 37 lengths need >= 19 chords (2 lengths each) + 40 rim.

    chords = []
    for d in range(2, 21):
        e = norm_edge(0, d)
        if e not in edges_set:
            edges_set.add(e)
            chords.append((0, d, d))

    # Build cycles for lengths 3..40 using the chord structure.
    # Chord (0,d): forward cycle = [0,1,...,d] (len d+1)
    #                backward cycle = [0,d,d+1,...,39] (len 41-d)
    all_cycles = {N: list(range(N))}  # length 40: full rim

    for d in range(2, 21):
        len_fwd = d + 1          # 3..21
        len_bwd = N - d + 1      # 39..21
        if len_fwd not in all_cycles:
            all_cycles[len_fwd] = list(range(0, d + 1))
        if len_bwd not in all_cycles:
            all_cycles[len_bwd] = [0] + list(range(d, N))

    # Verify all 38 cycles (lengths 3..40) are present and valid
    valid = True
    for k in range(3, N + 1):
        cyc = all_cycles.get(k)
        if cyc is None or len(cyc) != k or len(set(cyc)) != k:
            valid = False
            break
        for i in range(k):
            if norm_edge(cyc[i], cyc[(i + 1) % k]) not in edges_set:
                valid = False
                break
        if not valid:
            break

    if valid:
        return {'edges': [list(e) for e in edges_set],
                'cycles': [all_cycles[k] for k in range(3, N + 1)]}
    else:
        # Fallback: fan graph (77 edges, objective 37)
        fs = set()
        for i in range(1, N):
            fs.add(norm_edge(0, i))
        for i in range(1, N - 1):
            fs.add(norm_edge(i, i + 1))
        return {'edges': [list(e) for e in fs],
                'cycles': [list(range(k)) for k in range(3, N + 1)]}
