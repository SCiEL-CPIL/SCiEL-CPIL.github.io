import time
import random

N = 256

# ---- Precompute per-element 3-AP pair lists and 4-AP conflict triples ----
f3 = [[] for _ in range(N + 1)]   # x as FIRST term of a 3-AP: (x+d, x+2d)
m3 = [[] for _ in range(N + 1)]   # x as MIDDLE term: (x-d, x+d)
l3 = [[] for _ in range(N + 1)]   # x as LAST term: (x-2d, x-d)
f4 = [[] for _ in range(N + 1)]   # 4-APs containing x: tuple of the OTHER three elements

for x in range(1, N + 1):
    # x first: x+2d <= N
    for d in range(1, (N - x) // 2 + 1):
        f3[x].append((x + d, x + 2 * d))
    # x middle: x-d >= 1 and x+d <= N
    for d in range(1, min(x - 1, N - x) + 1):
        m3[x].append((x - d, x + d))
    # x last: x-2d >= 1
    for d in range(1, (x - 1) // 2 + 1):
        l3[x].append((x - 2 * d, x - d))
    # x in a 4-AP, four positions
    # pos 1: x, x+d, x+2d, x+3d
    for d in range(1, (N - x) // 3 + 1):
        f4[x].append((x + d, x + 2 * d, x + 3 * d))
    # pos 2: x-d, x, x+d, x+2d
    for d in range(1, min(x - 1, (N - x) // 2) + 1):
        f4[x].append((x - d, x + d, x + 2 * d))
    # pos 3: x-2d, x-d, x, x+d
    for d in range(1, min((x - 1) // 2, N - x) + 1):
        f4[x].append((x - 2 * d, x - d, x + d))
    # pos 4: x-3d, x-2d, x-d, x
    for d in range(1, (x - 1) // 3 + 1):
        f4[x].append((x - 3 * d, x - 2 * d, x - d))

# Precompute quadratic coefficients for incremental QUBO updates
# Q[x][y] = number of 3-APs that would be formed if both x and y are present
# (i.e., x and y complete a 3-AP with some third element z)
Q = [[0] * (N + 1) for _ in range(N + 1)]
for x in range(1, N + 1):
    # x as first: (x, x+d, x+2d)
    for (a, b) in f3[x]:
        Q[x][a] += 1
        Q[a][x] += 1
        Q[x][b] += 1
        Q[b][x] += 1
    # x as middle: (x-d, x, x+d)
    for (a, b) in m3[x]:
        Q[x][a] += 1
        Q[a][x] += 1
        Q[x][b] += 1
        Q[b][x] += 1
    # x as last: (x-2d, x-d, x)
    for (a, b) in l3[x]:
        Q[x][a] += 1
        Q[a][x] += 1
        Q[x][b] += 1
        Q[b][x] += 1

# Linear coefficients: L[x] = number of 3-APs where x is the middle term
# (these are "free" in the sense that they don't require a specific pair)
# Actually for incremental updates, we need:
# gain(x) = sum over y in present of Q[x][y]
# But we also need to account for 3-APs where x is the middle
# Let's redefine: 
# When adding x, the new 3-APs are those where x completes a triple with two existing elements
# This is: for each y in present, count how many z in present such that {x,y,z} forms a 3-AP
# This is equivalent to: sum_{y in present} (number of z in present such that x,y,z form 3-AP)
# But we can precompute for each pair (x,y) the number of z such that {x,y,z} is a 3-AP
# That's exactly Q[x][y] as defined above (but we need to be careful about double counting)

# Let's redefine Q more carefully:
# Q[x][y] = number of distinct z such that {x, y, z} forms a 3-AP with positive difference
# Each 3-AP is counted once per pair of its elements
# So the total number of 3-APs in a set S is: (1/3) * sum_{x,y in S} Q[x][y]
# Wait, that's not right. Each 3-AP has 3 pairs, so sum_{x,y in S} Q[x][y] counts each 3-AP exactly 3 times
# So total 3-APs = (1/3) * sum_{x,y in S, x!=y} Q[x][y]
# But Q[x][x] should be 0 (no self-pairs)

# Let's recompute Q properly
Q = [[0] * (N + 1) for _ in range(N + 1)]
for x in range(1, N + 1):
    # For each 3-AP containing x, increment Q for each pair
    # x as first: (x, x+d, x+2d)
    for (a, b) in f3[x]:
        Q[x][a] += 1
        Q[a][x] += 1
        Q[x][b] += 1
        Q[b][x] += 1
        Q[a][b] += 1
        Q[b][a] += 1
    # x as middle: (x-d, x, x+d)
    for (a, b) in m3[x]:
        Q[x][a] += 1
        Q[a][x] += 1
        Q[x][b] += 1
        Q[b][x] += 1
        Q[a][b] += 1
        Q[b][a] += 1
    # x as last: (x-2d, x-d, x)
    for (a, b) in l3[x]:
        Q[x][a] += 1
        Q[a][x] += 1
        Q[x][b] += 1
        Q[b][x] += 1
        Q[a][b] += 1
        Q[b][a] += 1

# Now Q[x][y] = number of z such that {x,y,z} is a 3-AP
# Total 3-APs in set S = (1/3) * sum_{x in S} sum_{y in S, y != x} Q[x][y]
# But since Q is symmetric, = (2/3) * sum_{x < y, x,y in S} Q[x][y]

# For incremental updates:
# When adding x to S, new 3-APs = sum_{y in S} Q[x][y] / 3 * 3 = sum_{y in S} Q[x][y]
# Wait, each new 3-AP involves x and two existing elements y,z
# The number of such 3-APs = sum_{y in S} (number of z in S such that {x,y,z} is 3-AP)
# But Q[x][y] counts z such that {x,y,z} is 3-AP, regardless of whether z is in S
# So we need: sum_{y in S} (number of z in S such that {x,y,z} is 3-AP)
# This is not simply sum_{y in S} Q[x][y]
# We need a different approach

# Let's use the original gain computation but make it faster
# gain(x) = number of 3-APs formed when x is added
# = (number of y,z in S such that {x,y,z} is 3-AP)
# We can compute this by checking all 3-APs containing x and seeing if the other two are in S

def count_3AP_total(present):
    s = 0
    for x in range(1, N + 1):
        if present[x]:
            for (a, b) in m3[x]:
                if present[a] and present[b]:
                    s += 1
    return s


def greedy_seed(rng):
    present = [False] * (N + 1)
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        ok = True
        for (a, b, c) in f4[x]:
            if present[a] and present[b] and present[c]:
                ok = False
                break
        if ok:
            present[x] = True
    return present


def tabu_search(present, deadline, rng):
    """Tabu search with incremental scoring for QUBO-like problem"""
    
    # Precompute for fast gain calculation
    # gain(x) = number of 3-APs formed when x is added to current set
    # We'll compute this on the fly using the precomputed lists
    
    def creates_4AP(x):
        for (a, b, c) in f4[x]:
            if present[a] and present[b] and present[c]:
                return True
        return False

    def gain_add(x):
        """Number of new 3-APs when adding x to current set"""
        g = 0
        # x as first: (x, x+d, x+2d) - need x+d and x+2d in set
        for (a, b) in f3[x]:
            if present[a] and present[b]:
                g += 1
        # x as middle: (x-d, x, x+d) - need x-d and x+d in set
        for (a, b) in m3[x]:
            if present[a] and present[b]:
                g += 1
        # x as last: (x-2d, x-d, x) - need x-2d and x-d in set
        for (a, b) in l3[x]:
            if present[a] and present[b]:
                g += 1
        return g

    def loss_remove(x):
        """Number of 3-APs lost when removing x from current set"""
        g = 0
        # x as first: (x, x+d, x+2d) - lose if x+d and x+2d in set
        for (a, b) in f3[x]:
            if present[a] and present[b]:
                g += 1
        # x as middle: (x-d, x, x+d) - lose if x-d and x+d in set
        for (a, b) in m3[x]:
            if present[a] and present[b]:
                g += 1
        # x as last: (x-2d, x-d, x) - lose if x-2d and x-d in set
        for (a, b) in l3[x]:
            if present[a] and present[b]:
                g += 1
        return g

    # Tabu list: (element, iteration_when_becomes_free)
    tabu = {}
    tabu_tenure = 5 + rng.randint(0, 5)
    current_iter = 0
    
    # Current objective
    current_score = count_3AP_total(present)
    best_score = current_score
    best_present = present[:]
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        current_iter += 1
        
        # Update tabu list
        to_remove = [k for k, v in tabu.items() if v <= current_iter]
        for k in to_remove:
            del tabu[k]
        
        # Generate moves: add x, remove x, or swap (remove r, add x)
        best_move = None
        best_gain = 0
        best_aspiration = False
        
        # Try additions
        for x in range(1, N + 1):
            if present[x]:
                continue
            if creates_4AP(x):
                continue
            if x in tabu and tabu[x] > current_iter:
                # Aspiration criterion: allow if it improves best
                g = gain_add(x)
                if g > best_gain - 1000:  # placeholder, will check aspiration properly
                    pass
                else:
                    continue
            g = gain_add(x)
            if g > best_gain:
                best_gain = g
                best_move = ('add', x)
        
        # Try removals (only if they enable better additions)
        # Actually, let's try swaps directly
        for r in range(1, N + 1):
            if not present[r]:
                continue
            loss = loss_remove(r)
            for x in range(1, N + 1):
                if present[x]:
                    continue
                # Temporarily remove r
                present[r] = False
                if not creates_4AP(x):
                    gain = gain_add(x)
                    net = gain - loss
                    if net > best_gain:
                        # Check tabu
                        is_tabu = (r in tabu and tabu[r] > current_iter) or (x in tabu and tabu[x] > current_iter)
                        if not is_tabu or net > best_score:  # aspiration
                            best_gain = net
                            best_move = ('swap', r, x)
                present[r] = True
        
        if best_move is None:
            # No improving move found, try random perturbation
            if no_improve_count >= max_no_improve:
                # Perturb: remove 2-3 random elements and add 2-3 random valid ones
                n_perturb = rng.randint(2, 3)
                to_remove = [x for x in range(1, N + 1) if present[x]]
                rng.shuffle(to_remove)
                to_remove = to_remove[:n_perturb]
                for r in to_remove:
                    present[r] = False
                    tabu[r] = current_iter + tabu_tenure
                
                to_add = [x for x in range(1, N + 1) if not present[x]]
                rng.shuffle(to_add)
                for x in to_add:
                    if creates_4AP(x):
                        continue
                    present[x] = True
                    tabu[x] = current_iter + tabu_tenure
                    if sum(1 for y in range(1, N + 1) if present[y]) >= len(to_remove):
                        break
                
                current_score = count_3AP_total(present)
                no_improve_count = 0
                continue
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve * 2:
                    break
                continue
        
        # Execute best move
        if best_move[0] == 'add':
            x = best_move[1]
            present[x] = True
            tabu[x] = current_iter + tabu_tenure
            current_score += best_gain
        elif best_move[0] == 'swap':
            r, x = best_move[1], best_move[2]
            present[r] = False
            present[x] = True
            tabu[r] = current_iter + tabu_tenure
            tabu[x] = current_iter + tabu_tenure
            current_score += best_gain
        
        if current_score > best_score:
            best_score = current_score
            best_present = present[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    # Restore best solution
    present[:] = best_present
    return best_score


def construct():
    start = time.time()
    # safety margin well under the 60s eval / 90s kill
    deadline = start + 52.0
    rng = random.Random(12345)

    best_present = None
    best_score = -1

    # main multi-restart search with tabu
    while time.time() < deadline - 4.0:
        present = greedy_seed(rng)
        score = tabu_search(present, deadline, rng)
        if score > best_score:
            best_score = score
            best_present = present[:]

    # final polish with remaining time on the incumbent
    if time.time() < deadline - 0.5 and best_present is not None:
        score = tabu_search(best_present, deadline, rng)
        if score > best_score:
            best_score = score

    if best_present is None:
        best_present = greedy_seed(random.Random(0))

    A = [x for x in range(1, N + 1) if best_present[x]]
    if not A:
        A = [1]

    # hard validity guarantee: drop any element that sits in a 4-AP
    present = [False] * (N + 1)
    for x in A:
        present[x] = True
    for x in list(A):
        for (a, b, c) in f4[x]:
            if present[a] and present[b] and present[c]:
                present[x] = False
                A.remove(x)
                break
    if not A:
        A = [1]
    return A
