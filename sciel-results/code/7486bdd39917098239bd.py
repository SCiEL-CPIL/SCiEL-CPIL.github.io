import random
import time
import numpy as np

N = 256

# Primitive polynomials over GF(2) of degree 8 (for 255-bit maximal sequences)
# x^8 + x^4 + x^3 + x^2 + 1, etc. We'll use several to diversify.
# Each is represented as a list of tap positions (0-indexed from LSB) including the feedback tap.
# For an LFSR of degree m with characteristic polynomial, taps are the exponents of the terms.
_PRIMITIVE_POLYS = [
    [0, 1, 4, 8],   # x^8 + x^4 + x^1 + x^0  (feedback from bit 7)
    [0, 2, 3, 8],   # x^8 + x^3 + x^2 + x^0
    [1, 4, 5, 8],   # x^8 + x^5 + x^4 + x^1
    [0, 3, 7, 8],   # x^8 + x^7 + x^3 + x^0
    [0, 1, 2, 8],   # x^8 + x^2 + x^1 + x^0
    [2, 3, 5, 8],   # x^8 + x^5 + x^3 + x^2
]

def _lfsr_sequence(poly_taps, init_state, length=256, phase=0):
    """
    Generate a length-bit sequence from an LFSR over GF(2).
    poly_taps: list of tap positions (integers), the highest is the degree.
    init_state: integer in [1, 2^degree - 1] (non-zero initial state).
    phase: skip this many bits before starting output (for diversity).
    """
    degree = max(poly_taps)
    mask = (1 << degree) - 1
    state = init_state & mask
    if state == 0:
        state = 1
    seq = []
    # Pre-generate phase + length bits
    total = phase + length
    bits = []
    for _ in range(total):
        # Feedback: XOR of tapped bits
        fb = 0
        for t in poly_taps:
            fb ^= (state >> t) & 1
        # Shift right, feedback goes into MSB (position degree-1)
        state = ((state >> 1) | (fb << (degree - 1))) & mask
        bits.append(state & 1)
    return bits[phase:phase + length]

def _build_aps():
    """Build list of all 4-term arithmetic progressions in 1..256 (0-indexed)."""
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _build_pos_to_aps(aps):
    """For each position, store (ap_index, position_in_ap) for all APs it belongs to."""
    pos_to_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_to_aps[a].append((idx, 0))
        pos_to_aps[b].append((idx, 1))
        pos_to_aps[c].append((idx, 2))
        pos_to_aps[d].append((idx, 3))
    return pos_to_aps

def _score_full(aps, colors):
    """Count monochromatic 4-APs (full recount)."""
    total = 0
    for a, b, c, d in aps:
        if colors[a] == colors[b] == colors[c] == colors[d]:
            total += 1
    return total

def _local_search_incremental(aps, pos_to_aps, colors, max_passes=200, deadline=None):
    """
    Greedy local search using incremental delta scoring.
    When flipping position i, only APs containing i need re-evaluation.
    """
    colors = list(colors)
    n = len(colors)

    # Current mono status of each AP
    mono_flags = np.zeros(len(aps), dtype=np.int8)
    for idx in range(len(aps)):
        a, b, c, d = aps[idx]
        mono_flags[idx] = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
    current_score = int(mono_flags.sum())

    for _pass in range(max_passes):
        if deadline and time.time() > deadline:
            break
        improved = False
        order = list(range(n))
        random.shuffle(order)
        for i in order:
            old = colors[i]
            new = 1 - old
            colors[i] = new

            delta = 0
            for ap_idx, pos_in_ap in pos_to_aps[i]:
                a, b, c, d = aps[ap_idx]
                was_mono = mono_flags[ap_idx]
                now_mono = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
                if was_mono and not now_mono:
                    delta -= 1
                elif not was_mono and now_mono:
                    delta += 1
                mono_flags[ap_idx] = now_mono

            if delta < 0:
                current_score += delta
                improved = True
            else:
                colors[i] = old
                for ap_idx, pos_in_ap in pos_to_aps[i]:
                    a, b, c, d = aps[ap_idx]
                    mono_flags[ap_idx] = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0

        if not improved:
            break

    return colors, current_score

def _polish_local_search(aps, pos_to_aps, colors, max_passes=1500, deadline=None):
    """
    High-precision polish: repeated multi-pass local search with random restarts
    of the scan order, to escape shallow local minima near the final solution.
    """
    colors = list(colors)
    n = len(colors)

    mono_flags = np.zeros(len(aps), dtype=np.int8)
    for idx in range(len(aps)):
        a, b, c, d = aps[idx]
        mono_flags[idx] = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
    current_score = int(mono_flags.sum())

    best_score = current_score
    best_colors = colors[:]

    for _pass in range(max_passes):
        if deadline and time.time() > deadline:
            break
        improved = False
        order = list(range(n))
        random.shuffle(order)
        for i in order:
            old = colors[i]
            new = 1 - old
            colors[i] = new

            delta = 0
            for ap_idx, pos_in_ap in pos_to_aps[i]:
                a, b, c, d = aps[ap_idx]
                was_mono = mono_flags[ap_idx]
                now_mono = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
                if was_mono and not now_mono:
                    delta -= 1
                elif not was_mono and now_mono:
                    delta += 1
                mono_flags[ap_idx] = now_mono

            if delta < 0:
                current_score += delta
                improved = True
            else:
                colors[i] = old
                for ap_idx, pos_in_ap in pos_to_aps[i]:
                    a, b, c, d = aps[ap_idx]
                    mono_flags[ap_idx] = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0

        if current_score < best_score:
            best_score = current_score
            best_colors = colors[:]

        if not improved:
            # No improvement this pass; try a few more passes with fresh order
            # before giving up, to escape potential order-dependent stalls.
            # If 3 consecutive no-improvement passes, stop.
            if _pass > 0 and not improved:
                # count consecutive; simple heuristic: stop after 3
                pass

    return best_colors, best_score

def _simulated_annealing(aps, pos_to_aps, colors, start_temp=10.0, end_temp=0.1, max_iter=5000, deadline=None):
    """
    Simulated annealing with single-bit and block-flip moves.
    """
    colors = list(colors)
    n = len(colors)

    mono_flags = np.zeros(len(aps), dtype=np.int8)
    for idx in range(len(aps)):
        a, b, c, d = aps[idx]
        mono_flags[idx] = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
    current_score = int(mono_flags.sum())
    best_score = current_score
    best_colors = colors[:]

    temp = start_temp
    cooling = (end_temp / start_temp) ** (1.0 / max_iter)

    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        temp *= cooling

        if random.random() < 0.7:
            i = random.randint(0, n - 1)
            old = colors[i]
            colors[i] = 1 - old

            delta = 0
            for ap_idx, pos_in_ap in pos_to_aps[i]:
                a, b, c, d = aps[ap_idx]
                was_mono = mono_flags[ap_idx]
                now_mono = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
                if was_mono and not now_mono:
                    delta -= 1
                elif not was_mono and now_mono:
                    delta += 1
                mono_flags[ap_idx] = now_mono

            new_score = current_score + delta
            if new_score < current_score or random.random() < np.exp(-delta / max(temp, 1e-10)):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
            else:
                colors[i] = old
                for ap_idx, pos_in_ap in pos_to_aps[i]:
                    a, b, c, d = aps[ap_idx]
                    mono_flags[ap_idx] = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
        else:
            length = random.randint(4, 16)
            start = random.randint(0, n - length)
            end = start + length

            old_segment = colors[start:end]
            old_mono = mono_flags.copy()

            colors[start:end] = old_segment[::-1]

            affected_aps = set()
            for i in range(start, end):
                for ap_idx, pos_in_ap in pos_to_aps[i]:
                    affected_aps.add(ap_idx)

            new_score = current_score
            for ap_idx in affected_aps:
                a, b, c, d = aps[ap_idx]
                was_mono = old_mono[ap_idx]
                now_mono = 1 if (colors[a] == colors[b] == colors[c] == colors[d]) else 0
                if was_mono and not now_mono:
                    new_score -= 1
                elif not was_mono and now_mono:
                    new_score += 1
                mono_flags[ap_idx] = now_mono

            delta = new_score - current_score
            if new_score < current_score or random.random() < np.exp(-delta / max(temp, 1e-10)):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
            else:
                colors[start:end] = old_segment
                mono_flags = old_mono

    return best_colors, best_score

def _lfsr_population_init(count):
    """Generate diverse LFSR-based colorings."""
    pops = []
    seen = set()
    for poly_idx, poly_taps in enumerate(_PRIMITIVE_POLYS):
        degree = max(poly_taps)
        max_state = (1 << degree) - 1
        # Try several initial states and phases
        for init in range(1, min(max_state, 30) + 1):
            for phase in [0, 1, 7, 13, 64]:
                seq = _lfsr_sequence(poly_taps, init, length=N, phase=phase)
                key = tuple(seq)
                if key not in seen:
                    seen.add(key)
                    pops.append(list(seq))
                if len(pops) >= count:
                    return pops
    # Pad with random if not enough
    while len(pops) < count:
        pops.append([random.randint(0, 1) for _ in range(N)])
    return pops[:count]

def _ga_population_init(pop_size):
    """Initialize diverse GA population: LFSR + structured + random."""
    population = []

    # LFSR-based structured colorings (long-range correlations)
    lfsr_pops = _lfsr_population_init(pop_size // 2)
    population.extend(lfsr_pops)

    # Block patterns
    for block_size in [2, 3, 4, 5, 8, 16, 32]:
        population.append([(i // block_size) % 2 for i in range(N)])

    # Thue-Morse
    population.append([bin(i).count('1') % 2 for i in range(N)])

    # Striped patterns
    for period in [3, 5, 7, 11, 13, 17]:
        pattern = []
        for i in range(N):
            pos_in_period = i % period
            pattern.append(0 if pos_in_period < period // 2 else 1)
        population.append(pattern)

    # Quadratic residue-like
    for p in [3, 5, 7, 11, 13]:
        pop = []
        for i in range(N):
            is_res = any((x * x) % p == i % p for x in range(p))
            pop.append(1 if is_res else 0)
        population.append(pop)

    # 2D grid-inspired
    grid_pop = []
    for i in range(N):
        row = i // 16
        col = i % 16
        grid_pop.append((row + col) % 2)
    population.append(grid_pop)

    grid_pop2 = []
    for i in range(N):
        row = i // 16
        col = i % 16
        grid_pop2.append(((row // 2) + (col // 2)) % 2)
    population.append(grid_pop2)

    # Pad with random
    while len(population) < pop_size:
        population.append([random.randint(0, 1) for _ in range(N)])

    return population[:pop_size]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    random.seed(42)

    aps = _build_aps()
    pos_to_aps = _build_pos_to_aps(aps)

    best_score = float('inf')
    best_pop = None

    pop_size = 48
    population = _ga_population_init(pop_size)

    # Evaluate initial population
    scores = [_score_full(aps, pop) for pop in population]
    order = sorted(range(len(population)), key=lambda i: scores[i])
    population = [population[i] for i in order]
    scores = [scores[i] for i in order]

    best_score = scores[0]
    best_pop = population[0][:]

    # Local search on top candidates
    for i in range(min(10, len(population))):
        if time.time() > deadline - 40:
            break
        improved_pop, s = _local_search_incremental(aps, pos_to_aps, population[i], max_passes=150, deadline=deadline - 40)
        if s < best_score:
            best_score = s
            best_pop = improved_pop[:]

    # Simulated annealing on the best solution
    if time.time() < deadline - 30:
        sa_pop, s = _simulated_annealing(aps, pos_to_aps, best_pop, start_temp=8.0, end_temp=0.05, max_iter=3000, deadline=deadline - 25)
        if s < best_score:
            best_score = s
            best_pop = sa_pop[:]

        if time.time() < deadline - 20:
            ls_pop, s = _local_search_incremental(aps, pos_to_aps, best_pop, max_passes=200, deadline=deadline - 15)
            if s < best_score:
                best_score = s
                best_pop = ls_pop[:]

    # Main GA loop
    mutation_rate = 0.06
    elitism = 3
    gen = 0

    while time.time() < deadline - 30:
        gen += 1

        def tournament_select():
            idx1 = random.randint(0, len(population) - 1)
            idx2 = random.randint(0, len(population) - 1)
            return population[idx1][:] if scores[idx1] <= scores[idx2] else population[idx2][:]

        new_population = []

        for i in range(elitism):
            new_population.append(population[i][:])

        while len(new_population) < pop_size:
            if time.time() > deadline - 25:
                break
            p1 = tournament_select()
            p2 = tournament_select()

            child = [p1[i] if random.random() < 0.5 else p2[i] for i in range(N)]

            for i in range(N):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]

            new_population.append(child)

        new_scores = [_score_full(aps, pop) for pop in new_population]

        order = sorted(range(len(new_population)), key=lambda i: new_scores[i])
        population = [new_population[i] for i in order]
        scores = [new_scores[i] for i in order]

        if scores[0] < best_score:
            best_score = scores[0]
            best_pop = population[0][:]

        for i in range(min(4, len(population))):
            if time.time() > deadline - 20:
                break
            improved_pop, s = _local_search_incremental(aps, pos_to_aps, population[i], max_passes=80, deadline=deadline - 20)
            if s < best_score:
                best_score = s
                best_pop = improved_pop[:]

        # Periodically inject LFSR diversity
        if gen % 6 == 0 and time.time() < deadline - 35:
            for _ in range(4):
                new_ind = _lfsr_population_init(1)[0]
                s = _score_full(aps, new_ind)
                if s < scores[-1]:
                    population[-1] = new_ind
                    scores[-1] = s

    # Final SA polish
    if time.time() < deadline - 15:
        sa_pop, s = _simulated_annealing(aps, pos_to_aps, best_pop, start_temp=2.0, end_temp=0.01, max_iter=2000, deadline=deadline - 12)
        if s < best_score:
            best_score = s
            best_pop = sa_pop[:]

    # Final high-precision local search polish
    if time.time() < deadline - 5:
        final_pop, s = _polish_local_search(aps, pos_to_aps, best_pop, max_passes=800, deadline=deadline - 2)
        if s < best_score:
            best_score = s
            best_pop = final_pop[:]

    # Last local search
    if time.time() < deadline - 1:
        final_pop, s = _local_search_incremental(aps, pos_to_aps, best_pop, max_passes=150, deadline=deadline - 0.5)
        if s < best_score:
            best_score = s
            best_pop = final_pop[:]

    return best_pop
