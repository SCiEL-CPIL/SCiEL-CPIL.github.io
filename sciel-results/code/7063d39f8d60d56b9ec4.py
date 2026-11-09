"""Covering 1..L by prime residue classes with reciprocal budget 1 (Simulated Annealing)."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

def construct():
    t0 = time.time()
    DEADLINE = 145.0
    MAXL = 100000
    primes = _primes_upto(1000)
    recip = {p: Fraction(1, p) for p in primes}
    budget0 = Fraction(1, 1)

    class Tracker:
        __slots__ = ("maxl", "count", "L", "state", "budget")
        def __init__(self, maxl):
            self.maxl = maxl
            self.count = [0] * (maxl + 1)
            self.L = 0
            self.state = {}
            self.budget = budget0

        def _start(self, p, r):
            return r if r >= 1 else p

        def add(self, p, r):
            if p in self.state:
                return False
            if self.budget < recip[p]:
                return False
            start = self._start(p, r)
            for x in range(start, self.maxl + 1, p):
                self.count[x] += 1
            self.state[p] = r
            self.budget -= recip[p]
            self._extend_L()
            return True

        def remove(self, p):
            if p not in self.state:
                return False
            r = self.state[p]
            start = self._start(p, r)
            for x in range(start, self.maxl + 1, p):
                self.count[x] -= 1
            del self.state[p]
            self.budget += recip[p]
            self._recompute_L()
            return True

        def change_residue(self, p, new_r):
            if p not in self.state:
                return False
            old_r = self.state[p]
            if old_r == new_r:
                return False
            start_old = self._start(p, old_r)
            for x in range(start_old, self.maxl + 1, p):
                self.count[x] -= 1
            start_new = self._start(p, new_r)
            for x in range(start_new, self.maxl + 1, p):
                self.count[x] += 1
            self.state[p] = new_r
            self._recompute_L()
            return True

        def _extend_L(self):
            newL = self.L
            for i in range(self.L + 1, self.maxl + 1):
                if self.count[i] > 0:
                    newL = i
                else:
                    break
            self.L = newL

        def _recompute_L(self):
            newL = 0
            for i in range(1, self.maxl + 1):
                if self.count[i] > 0:
                    newL = i
                else:
                    break
            self.L = newL

        def first_gap(self):
            for i in range(self.L + 1, self.maxl + 1):
                if self.count[i] == 0:
                    return i
            return None

    def greedy_init(seed):
        rng = random.Random(seed)
        tr = Tracker(MAXL)
        order = primes[:]
        rng.shuffle(order)
        
        for _ in range(len(order)):
            if time.time() > DEADLINE:
                break
            gap = tr.first_gap()
            if gap is None:
                break
                
            best_gain = 0
            best_p = None
            best_r = None
            
            for p in order:
                if p in tr.state or tr.budget < recip[p]:
                    continue
                r = gap % p
                start = r if r >= 1 else p
                gain = 0
                limit = min(gap + 100, MAXL)
                for x in range(gap, limit + 1):
                    if tr.count[x] == 0 and (x - start) % p == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = r
            
            if best_p is None:
                break
            tr.add(best_p, best_r)
            
        return tr

    best_tr = None
    best_L = 0
    for seed in range(5):
        if time.time() > DEADLINE:
            break
        tr = greedy_init(seed)
        if tr.L > best_L:
            best_L = tr.L
            best_tr = tr

    if best_tr is None:
        return [[p, 1] for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]]

    tr = best_tr
    best_state = tr.state.copy()
    best_L_found = tr.L

    T = 100.0
    T_min = 0.01
    alpha = 0.999
    rng = random.Random(42)
    prime_list = primes

    while time.time() < DEADLINE:
        if T < T_min:
            break

        move_type = rng.random()
        
        if move_type < 0.5:
            # Residue change
            if not tr.state:
                continue
            p = rng.choice(list(tr.state.keys()))
            old_r = tr.state[p]
            new_r = rng.randrange(p)
            if new_r == old_r:
                continue
            old_L = tr.L
            tr.change_residue(p, new_r)
            new_L = tr.L
            delta = new_L - old_L
            if delta <= 0:
                prob = 0.0
                if delta < 0 and T > 0:
                    exponent = -(-delta) / T
                    if exponent < -700:
                        prob = 0.0
                    else:
                        prob = 2.718281828 ** exponent
                if rng.random() > prob:
                    tr.change_residue(p, old_r)

        elif move_type < 0.8:
            # Prime Swap
            if not tr.state:
                continue
            p_out = rng.choice(list(tr.state.keys()))
            r_out = tr.state[p_out]
            candidates = [p for p in prime_list if p not in tr.state]
            if not candidates:
                continue
            p_in = rng.choice(candidates)
            
            if tr.budget + recip[p_out] - recip[p_in] < 0:
                continue
            
            gap = tr.first_gap()
            r_in = 0 if gap is None else gap % p_in
            
            old_L = tr.L
            tr.remove(p_out)
            tr.add(p_in, r_in)
            new_L = tr.L
            delta = new_L - old_L
            
            if delta <= 0:
                prob = 0.0
                if delta < 0 and T > 0:
                    exponent = -(-delta) / T
                    if exponent < -700:
                        prob = 0.0
                    else:
                        prob = 2.718281828 ** exponent
                if rng.random() > prob:
                    tr.remove(p_in)
                    tr.add(p_out, r_out)

        else:
            # Prime Add
            candidates = [p for p in prime_list if p not in tr.state]
            if not candidates:
                continue
            p_in = rng.choice(candidates)
            if tr.budget < recip[p_in]:
                continue
            
            gap = tr.first_gap()
            r_in = 0 if gap is None else gap % p_in
            
            old_L = tr.L
            tr.add(p_in, r_in)
            new_L = tr.L
            delta = new_L - old_L
            
            if delta <= 0:
                prob = 0.0
                if delta < 0 and T > 0:
                    exponent = -(-delta) / T
                    if exponent < -700:
                        prob = 0.0
                    else:
                        prob = 2.718281828 ** exponent
                if rng.random() > prob:
                    tr.remove(p_in)

        if tr.L > best_L_found:
            best_L_found = tr.L
            best_state = tr.state.copy()
            
        T *= alpha

    result = [[p, r] for p, r in best_state.items()]
    return result
# EVOLVE-BLOCK-END
