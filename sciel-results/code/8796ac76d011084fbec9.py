import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_n = 0
    best_colors = None
    
    # Try decreasing n
    for n in [34, 32, 30, 28, 25, 20, 15, 10]:
        if time.time() > deadline - 5:
            break
            
        result = solve_for_n(n, deadline - 5)
        if result:
            # Since we go downwards, the first success is the max
            # But if we fail on 34, we try 32.
            # So we return the first success.
            return result
            
    # Fallback
    return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}

def solve_for_n(n, deadline):
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    trip_to_idx = {t: i for i, t in enumerate(triples)}
    
    k4s = [[] for _ in range(num_triples)]
    k5s = [[] for _ in range(num_triples)]
    other_k4 = [{} for _ in range(num_triples)]
    other_k5 = [{} for _ in range(num_triples)]
    
    for i, t in enumerate(triples):
        a, b, c = t
        for d in range(n):
            if d in (a, b, c):
                continue
            t1 = tuple(sorted((a, b, d)))
            t2 = tuple(sorted((a, c, d)))
            t3 = tuple(sorted((b, c, d)))
            idx1 = trip_to_idx[t1]
            idx2 = trip_to_idx[t2]
            idx3 = trip_to_idx[t3]
            k4s[i].append(d)
            other_k4[i][d] = (idx1, idx2, idx3)
            
        for d in range(n):
            if d in (a, b, c):
                continue
            for e in range(n):
                if e in (a, b, c, d) or e <= d:
                    continue
                verts = [a, b, c, d, e]
                all_trips = list(combinations(verts, 3))
                other_idxs = []
                for tt in all_trips:
                    if tt != t:
                        other_idxs.append(trip_to_idx[tt])
                k5s[i].append((d, e))
                other_k5[i][(d, e)] = tuple(other_idxs)

    def calc_cost(cols):
        red_k4 = 0
        blue_k5 = 0
        for k4 in combinations(range(n), 4):
            a, b, c, d = k4
            t1 = trip_to_idx[(a,b,c)]
            t2 = trip_to_idx[(a,b,d)]
            t3 = trip_to_idx[(a,c,d)]
            t4 = trip_to_idx[(b,c,d)]
            if cols[t1] == 1 and cols[t2] == 1 and cols[t3] == 1 and cols[t4] == 1:
                red_k4 += 1
        for k5 in combinations(range(n), 5):
            a, b, c, d, e = k5
            is_blue = True
            for tt in combinations(k5, 3):
                if cols[trip_to_idx[tt]] == 1:
                    is_blue = False
                    break
            if is_blue:
                blue_k5 += 1
        return red_k4 + blue_k5

    def get_delta(t_idx, new_color):
        old_color = colors[t_idx]
        if old_color == new_color:
            return 0
        delta = 0
        for d in k4s[t_idx]:
            idx1, idx2, idx3 = other_k4[t_idx][d]
            if old_color == 1:
                if colors[idx1] == 1 and colors[idx2] == 1 and colors[idx3] == 1:
                    delta -= 1
            else:
                if colors[idx1] == 1 and colors[idx2] == 1 and colors[idx3] == 1:
                    delta += 1
        for d, e in k5s[t_idx]:
            idxs = other_k5[t_idx][(d, e)]
            if old_color == 0:
                all_blue = True
                for idx in idxs:
                    if colors[idx] != 0:
                        all_blue = False
                        break
                if all_blue:
                    delta -= 1
            else:
                all_blue = True
                for idx in idxs:
                    if colors[idx] != 0:
                        all_blue = False
                        break
                if all_blue:
                    delta += 1
        return delta

    # Search
    max_restarts = 20
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        current_cost = calc_cost(colors)
        if current_cost == 0:
            return {'n': n, 'colors': colors}
        
        stuck = 0
        while time.time() < deadline:
            t_idx = random.randint(0, num_triples - 1)
            new_color = 1 - colors[t_idx]
            delta = get_delta(t_idx, new_color)
            if delta < 0:
                colors[t_idx] = new_color
                current_cost += delta
                stuck = 0
                if current_cost == 0:
                    return {'n': n, 'colors': colors}
            elif delta == 0 and random.random() < 0.5:
                colors[t_idx] = new_color
                stuck = 0
            else:
                stuck += 1
            if stuck > 500:
                break
    return None
