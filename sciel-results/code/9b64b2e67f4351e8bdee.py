import time
import random

N = 51975

# Precompute all odd divisors of N greater than 1 (each usable at most once).
_DIVS = [d for d in range(3, N + 1, 2) if N % d == 0]
# For each modulus, a bitset of its residue classes: bit i is 1 iff i % m == r.
_BITSETS = {}
for _m in _DIVS:
    _bs = 0
    _r = 0
    for _i in range(_r, N, _m):
        _bs |= 1 << _i
    _BITSETS[_m] = _bs

# Full universe of residues.
_FULL = (1 << N) - 1


def construct():
    start = time.time()
    deadline = start + 147.0  # hard guard; evaluator kills at 150s

    def elapsed():
        return time.time() - start

    def covered_count(union):
        # number of residues covered by at least one selected class
        return (N - union).bit_count()

    def greedy_init():
        union = 0
        classes = []
        avail = list(_DIVS)
        while avail:
            if elapsed() > deadline:
                break
            best_gain = 0
            best_i = -1
            best_r = 0
            for i, m in enumerate(avail):
                # best residue for this modulus: maximizes new coverage
                gain = 0
                for r in range(m):
                    g = (_BITSETS[m] >> r).bit_count()
                    if g > gain:
                        gain = g
                        best_r = r
                # combine with current coverage for true marginal gain
                g = (_BITSETS[m] >> best_r & ~union).bit_count()
                if g > best_gain:
                    best_gain = g
                    best_i = i
                    best_r = r
            if best_i < 0 or best_gain == 0:
                break
            m = avail.pop(best_i)
            r = best_r
            classes.append([m, r])
            union |= _BITSETS[m] >> r
        return classes, union

    def local_search(classes, union):
        # Deterministic best-improvement polish until no improvement.
        improved = True
        while improved:
            if elapsed() > deadline:
                return classes, union
            improved = False
            # (1) best residue for each class
            for i in range(len(classes)):
                m, r = classes[i]
                base = union & ~(_BITSETS[m] >> r)
                best_r = r
                best_g = 0
                for rr in range(m):
                    g = (_BITSETS[m] >> rr & ~base).bit_count()
                    if g > best_g:
                        best_g = g
                        best_r = rr
                if best_r != r and best_g > 0:
                    union = base | (_BITSETS[m] >> best_r)
                    classes[i][1] = best_r
                    improved = True
            if not improved:
                # (2) best single swap (replace a class with an unused modulus)
                used = set(c[0] for c in classes)
                best_gain = 0
                best_op = None
                for i in range(len(classes)):
                    m_out, r_out = classes[i]
                    base = union & ~(_BITSETS[m_out] >> r_out)
                    for m_in in _DIVS:
                        if m_in in used:
                            continue
                        # best residue of m_in against base
                        bg = 0
                        br = 0
                        for rr in range(m_in):
                            g = (_BITSETS[m_in] >> rr & ~base).bit_count()
                            if g > bg:
                                bg = g
                                br = rr
                        if bg > best_gain:
                            best_gain = bg
                            best_op = ('swap', i, m_out, m_in, br)
                if best_op is not None and best_gain > 0:
                    _, i, m_out, m_in, br = best_op
                    m_out, r_out = classes[i]
                    base = union & ~(_BITSETS[m_out] >> r_out)
                    union = base | (_BITSETS[m_in] >> br)
                    classes[i] = [m_in, br]
                    improved = True
                else:
                    # (3) best single add (if any unused modulus helps)
                    for m_in in _DIVS:
                        if m_in in used:
                            continue
                        bg = 0
                        br = 0
                        for rr in range(m_in):
                            g = (_BITSETS[m_in] >> rr & ~union).bit_count()
                            if g > bg:
                                bg = g
                                br = rr
                        if bg > best_gain:
                            best_gain = bg
                            best_op = ('add', m_in, br)
                    if best_op is not None and best_gain > 0:
                        _, m_in, br = best_op
                        union |= _BITSETS[m_in] >> br
                        classes.append([m_in, br])
                        improved = True
        return classes, union

    def anneal(classes, union, T0=2.0, min_T=0.05, rate=0.9995):
        T = T0
        used = set(c[0] for c in classes)
        while T > min_T:
            if elapsed() > deadline:
                return classes, union
            mt = random.random()
            if mt < 0.45 and classes:
                # change residue of a random class
                i = random.randrange(len(classes))
                m, r = classes[i]
                r2 = random.randrange(m)
                if r2 == r:
                    T *= rate
                    continue
                base = union & ~(_BITSETS[m] >> r)
                delta = base.bit_count() - (base | (_BITSETS[m] >> r2)).bit_count()
                if delta < 0 or random.random() < 2.718281828459045 ** (-delta / T):
                    union = base | (_BITSETS[m] >> r2)
                    classes[i][1] = r2
            elif mt < 0.75 and classes:
                # swap a class for a random unused modulus
                i = random.randrange(len(classes))
                m_out, r_out = classes[i]
                unused = [m for m in _DIVS if m not in used]
                if not unused:
                    T *= rate
                    continue
                m_in = random.choice(unused)
                r_in = random.randrange(m_in)
                base = union & ~(_BITSETS[m_out] >> r_out)
                delta = base.bit_count() - (base | (_BITSETS[m_in] >> r_in)).bit_count()
                if delta < 0 or random.random() < 2.718281828459045 ** (-delta / T):
                    union = base | (_BITSETS[m_in] >> r_in)
                    classes[i] = [m_in, r_in]
                    used.discard(m_out)
                    used.add(m_in)
            elif mt < 0.90:
                # add a random unused modulus
                unused = [m for m in _DIVS if m not in used]
                if not unused:
                    T *= rate
                    continue
                m = random.choice(unused)
                r = random.randrange(m)
                delta = -(_BITSETS[m] >> r & ~union).bit_count()
                if delta < 0 or random.random() < 2.718281828459045 ** (-delta / T):
                    union |= _BITSETS[m] >> r
                    classes.append([m, r])
                    used.add(m)
            else:
                # remove a random class
                if not classes:
                    T *= rate
                    continue
                i = random.randrange(len(classes))
                m, r = classes[i]
                delta = (union & ~(_BITSETS[m] >> r)).bit_count() - union.bit_count()
                if delta < 0 or random.random() < 2.718281828459045 ** (-delta / T):
                    union &= ~(_BITSETS[m] >> r)
                    classes.pop(i)
                    used.discard(m)
            T *= rate
        return classes, union

    best_classes, best_union = greedy_init()
    best_classes, best_union = local_search(best_classes, best_union)
    best_score = covered_count(best_union)

    # A few SA restarts to escape local optima, always keeping the best.
    while elapsed() < deadline - 4.0:
        T0 = 2.0 + 0.3 * (random.random() - 0.5)
        c, u = greedy_init()
        c, u = local_search(c, u)
        c, u = anneal(c, u, T0=T0)
        c, u = local_search(c, u)
        s = covered_count(u)
        if s < best_score:
            best_score = s
            best_classes, best_union = c, u
            if best_score == 0:
                break

    return best_classes
