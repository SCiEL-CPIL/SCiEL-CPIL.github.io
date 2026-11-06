import math
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    N = 512
    # Precompute weights
    w = {a: Fraction(1, N - a) for a in range(1, N)}

    def score(s):
        return sum(w[a] for a in s)

    # Greedy initialization: pick largest coprime set
    chosen = set()
    for a in range(511, 0, -1):
        if all(math.gcd(a, b) == 1 for b in chosen):
            chosen.add(a)

    best = set(chosen)
    best_score = score(chosen)

    def try_move(current, score_val):
        """Try 1-swap and 2-swap moves, return best improving move."""
        cur = set(current)
        best_gain = Fraction(0)
        best_move = None

        # 1-swap: remove x in cur, add y not in cur, y coprime to cur\{x}
        for x in list(cur):
            rest = cur - {x}
            for y in range(511, 0, -1):
                if y in cur:
                    continue
                if y <= x:
                    continue
                if all(math.gcd(y, z) == 1 for z in rest):
                    gain = w[y] - w[x]
                    if gain > best_gain:
                        best_gain = gain
                        best_move = ('swap1', x, y)

        # 2-swap: remove x1,x2 in cur, add y1,y2 not in cur
        # y1,y2 coprime to cur\{x1,x2} and coprime to each other
        if elapsed() > deadline - 10:
            return best_move, best_gain

        cur_list = list(cur)
        for i in range(len(cur_list)):
            for j in range(i + 1, len(cur_list)):
                x1, x2 = cur_list[i], cur_list[j]
                rest = cur - {x1, x2}
                # candidate y's: must be coprime to rest
                cands = [y for y in range(511, 0, -1)
                         if y not in cur and y > max(x1, x2)
                         and all(math.gcd(y, z) == 1 for z in rest)]
                if len(cands) < 2:
                    continue
                # find best pair from cands
                for a_i in range(len(cands)):
                    for b_i in range(a_i + 1, len(cands)):
                        y1, y2 = cands[a_i], cands[b_i]
                        if math.gcd(y1, y2) != 1:
                            continue
                        gain = w[y1] + w[y2] - w[x1] - w[x2]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = ('swap2', x1, x2, y1, y2)
                if elapsed() > deadline - 10:
                    break
            if elapsed() > deadline - 10:
                break

        return best_move, best_gain

    improved = True
    while improved and elapsed() < deadline - 5:
        improved = False
        move, gain = try_move(best, best_score)
        if move is not None and gain > 0:
            if move[0] == 'swap1':
                _, x, y = move
                best.remove(x)
                best.add(y)
            else:
                _, x1, x2, y1, y2 = move
                best.remove(x1)
                best.remove(x2)
                best.add(y1)
                best.add(y2)
            best_score = score(best)
            improved = True

    # Simulated annealing polish
    if elapsed() < deadline - 8:
        current = set(best)
        current_score = best_score
        T = Fraction(1, 1000)
        T_min = Fraction(1, 1000000)
        cooling = 0.995
        while T > T_min and elapsed() < deadline - 3:
            # random 1-swap
            cur_list = list(current)
            x = random.choice(cur_list)
            rest = current - {x}
            # pick random y coprime to rest
            cands = [y for y in range(1, 512)
                     if y not in current and all(math.gcd(y, z) == 1 for z in rest)]
            if not cands:
                T *= cooling
                continue
            y = random.choice(cands)
            new_set = set(current)
            new_set.remove(x)
            new_set.add(y)
            new_score = score(new_set)
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(float(delta) / float(T)):
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best = set(current)
                    best_score = current_score
            T *= cooling

    return list(best)
