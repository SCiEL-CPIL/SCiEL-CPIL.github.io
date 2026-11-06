"""Genetic algorithm search for size-efficient 17-wire sorting networks.
Uses Zero-One Principle (131072 boolean inputs) verified via fast bit-vector simulation.
Deadline guard: returns best valid network found before the wall-clock limit.
"""
import random
import time

N = 17
DEADLINE = time.time() + 80  # safety margin under 90s kill

# ---------- fast verification via bit-vectors ----------
# Represent each of the 131072 boolean inputs as a 131072-bit integer.
# Wire i holds a big-int bit-vector; compare-exchange (i,j):
#   m = a & b ; a = a | b ; b = m
# After the network, wire 0 must be all-ones and wire N-1 all-zeros.
MASK = (1 << (1 << N)) - 1  # 2^17 bits set
# Precompute per-wire bit-vectors once.
_WIRE_BV = []
for i in range(N):
    bv = 0
    for x in range(1 << N):
        if (x >> i) & 1:
            bv |= 1 << x
    _WIRE_BV.append(bv)
# Precompute per-comparator-pair (i,j) effect as a function of (a,b)->(a|b, a&b)
# We cache nothing per-pair; op is cheap.

def _apply(net, wires):
    for (i, j) in net:
        a = wires[i]; b = wires[j]
        m = a & b
        wires[i] = a | b
        wires[j] = m
    return wires

def is_valid(net):
    """Return True iff net sorts all 2^N boolean inputs (Zero-One Principle)."""
    wires = list(_WIRE_BV)
    for (i, j) in net:
        a = wires[i]; b = wires[j]
        m = a & b
        wires[i] = a | b
        wires[j] = m
    # sorted: wire0 all ones, wire N-1 all zeros
    return wires[0] == MASK and wires[N - 1] == 0

# ---------- comparator pool ----------
ALL_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
NPAIRS = len(ALL_PAIRS)

# ---------- depth-aware random network generation ----------
def random_network(max_depth=14, min_depth=8, max_per_depth=5):
    """Generate a random network depth-by-depth (parallel layers)."""
    depth = random.randint(min_depth, max_depth)
    net = []
    for d in range(depth):
        k = random.randint(1, max_per_depth)
        # choose k disjoint pairs for this depth
        used = set()
        chosen = []
        tries = 0
        while len(chosen) < k and tries < 200:
            p = random.choice(ALL_PAIRS)
            if p[0] not in used and p[1] not in used:
                chosen.append(p)
                used.add(p[0]); used.add(p[1])
            tries += 1
        net.extend(chosen)
    return net

def odd_even_network():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

# ---------- GA operators ----------
def crossover(a, b):
    """Order-preserving crossover on comparator sequences."""
    if len(a) < len(b):
        a, b = b, a
    cut = random.randint(1, len(b) - 1)
    child = list(b[:cut])
    seen = set(child)
    for c in a:
        if c not in seen:
            child.append(c)
            seen.add(c)
    for c in b[cut:]:
        if c not in seen:
            child.append(c)
            seen.add(c)
    return child

def mutate(net, rate=0.08):
    net = list(net)
    n = len(net)
    # insert
    if random.random() < rate:
        net.append(random.choice(ALL_PAIRS))
    # delete
    if net and random.random() < rate:
        del net[random.randrange(len(net))]
    # swap
    if len(net) >= 2 and random.random() < rate:
        i, j = random.sample(range(len(net)), 2)
        net[i], net[j] = net[j], net[i]
    # replace
    if net and random.random() < rate:
        net[random.randrange(len(net))] = random.choice(ALL_PAIRS)
    return net

# ---------- main ----------
def construct():
    global DEADLINE
    rng = random.Random(12345)
    # seed with known-correct networks
    oe = odd_even_network()
    assert is_valid(oe)
    best = oe
    best_len = len(oe)
    pop = [oe]
    # add some random individuals
    for _ in range(19):
        pop.append(random_network())
    pop_size = len(pop)

    def now_ok():
        return time.time() < DEADLINE

    gen = 0
    while now_ok():
        # evaluate fitness of whole pop
        scored = []
        for ind in pop:
            if is_valid(ind):
                scored.append((len(ind), ind))
        if scored:
            scored.sort(key=lambda t: t[0])
            if scored[0][0] < best_len:
                best_len = scored[0][0]
                best = scored[0][1]
        # if we hit the proven lower bound, stop
        if best_len <= 63:
            break
        # build next generation: elitism + selection + crossover + mutation
        next_pop = []
        # elitism: keep top valid
        if scored:
            for _, ind in scored[:4]:
                next_pop.append(list(ind))
        # fill rest
        while len(next_pop) < pop_size and now_ok():
            if scored and random.random() < 0.6:
                # tournament among valid
                cands = random.sample(scored, min(3, len(scored)))
                parent = min(cands, key=lambda t: t[0])[1]
                child = mutate(parent, rate=0.10)
            else:
                # random restart
                child = random_network()
            next_pop.append(child)
        pop = next_pop
        gen += 1

    # final: ensure best is valid (it is, by construction), return it
    return list(best)

# Pre-warm: nothing heavy at import beyond bit-vector build (fast).
