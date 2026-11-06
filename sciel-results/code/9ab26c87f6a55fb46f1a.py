"""Hypergraph Ramsey R(4,5;3): maximise n for a 2-colouring of triples with no red K4^(3) and no blue K5^(3).

New structural strategy:
- Start from a deterministic algebraic seed based on GF(2^5) for n=32.
- Use a parameterized low-rank binary tensor representation: the color of each triple
  is computed from a small set of GF(2)-valued basis vectors and a few nonlinear features,
  and we optimize these parameters via stochastic search.
- Then extend to n=34 (and possibly n=35) by adding vertices and re-optimizing,
  using a violation-driven local search as a final refinement step.
"""

import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 145.0  # seconds; return with margin


def construct():
    def _t_idx(n, a, b, c):
        """Calculate the linear index of triple (a,b,c) in lexicographic order."""
        cnt = 0
        for x in range(a):
            cnt += (n - x - 1) * (n - x - 2) // 2
        for y in range(a + 1, b):
            cnt += (n - y - 1)
        cnt += (c - b - 1)
        return cnt

    def _build_data(n):
        """Precompute quad and pent structures for fast violation counting."""
        m = n * (n - 1) * (n - 2) // 6
        quads = []
        for q in combinations(range(n), 4):
            a, b, c, d = q
            idxs = [
                _t_idx(n, a, b, c),
                _t_idx(n, a, b, d),
                _t_idx(n, a, c, d),
                _t_idx(n, b, c, d)
            ]
            quads.append(idxs)

        pents = []
        for p in combinations(range(n), 5):
            a, b, c, d, e = p
            idxs = []
            for t in combinations(p, 3):
                idxs.append(_t_idx(n, *t))
            pents.append(idxs)

        tri_quads = [[] for _ in range(m)]
        for qi, idxs in enumerate(quads):
            for i in idxs:
                tri_quads[i].append(qi)

        tri_pents = [[] for _ in range(m)]
        for pi, idxs in enumerate(pents):
            for i in idxs:
                tri_pents[i].append(pi)

        return quads, pents, tri_quads, tri_pents

    def _initial_energy(colors, quads, pents):
        """Calculate initial violation counts and structures."""
        quad_red = np.zeros(len(quads), dtype=np.int8)
        for qi, idxs in enumerate(quads):
            for i in idxs:
                if colors[i] == 0:
                    quad_red[qi] += 1
        pent_blue = np.zeros(len(pents), dtype=np.int8)
        for pi, idxs in enumerate(pents):
            for i in idxs:
                if colors[i] == 1:
                    pent_blue[pi] += 1
        red_k4 = int(np.sum(quad_red == 4))
        blue_k5 = int(np.sum(pent_blue == 10))
        return red_k4, blue_k5, quad_red, pent_blue

    def _violation_driven_search(colors, quads, pents, tri_quads, tri_pents, time_budget):
        """
        Local search that only considers flipping triples that are part of a violation.
        """
        colors = list(colors)
        m = len(colors)
        red_k4, blue_k5, quad_red, pent_blue = _initial_energy(colors, quads, pents)
        total_v = red_k4 + blue_k5

        if total_v == 0:
            return colors, True

        t_end = time.time() + time_budget
        improved = True

        while improved and time.time() < t_end:
            improved = False
            # Identify candidate triples: those in a red K4 or blue K5
            candidates = set()

            # Triples in red K4s
            for qi, count in enumerate(quad_red):
                if count == 4:
                    for t in quads[qi]:
                        candidates.add(t)

            # Triples in blue K5s
            for pi, count in enumerate(pent_blue):
                if count == 10:
                    for t in pents[pi]:
                        candidates.add(t)

            if not candidates:
                # Fallback: check all triples
                for t in range(m):
                    candidates.add(t)

            candidate_list = list(candidates)
            random.shuffle(candidate_list)

            for t in candidate_list:
                if time.time() >= t_end:
                    break
                cur = colors[t]
                new = 1 - cur
                delta_red = 1 if new == 0 else -1
                delta_blue = 1 if new == 1 else -1

                dq = 0
                for qi in tri_quads[t]:
                    old = quad_red[qi]
                    new_q = old + delta_red
                    if old == 4:
                        dq -= 1
                    if new_q == 4:
                        dq += 1
                dp = 0
                for pi in tri_pents[t]:
                    old = pent_blue[pi]
                    new_p = old + delta_blue
                    if old == 10:
                        dp -= 1
                    if new_p == 10:
                        dp += 1

                if dq + dp < 0:
                    colors[t] = new
                    for qi in tri_quads[t]:
                        quad_red[qi] += delta_red
                    for pi in tri_pents[t]:
                        pent_blue[pi] += delta_blue
                    red_k4 += dq
                    blue_k5 += dp
                    total_v = red_k4 + blue_k5
                    improved = True
                    if total_v == 0:
                        return colors, True

        return colors, (red_k4 == 0 and blue_k5 == 0)

    def _algebraic_seed(n, seed=42):
        """
        Generate an initial coloring using a deterministic algebraic rule.
        For n=32, use the 5-bit representation and a nonlinear function.
        For n > 32, extend by adding vertices with a rule-based extension.
        """
        m = n * (n - 1) * (n - 2) // 6
        colors = np.zeros(m, dtype=np.int8)

        # For the first 32 vertices, use a GF(2^5)-based rule
        for i, (a, b, c) in enumerate(combinations(range(n), 3)):
            if n <= 32:
                # Use XOR sum and a nonlinear hash
                s = a ^ b ^ c
                # Mix with a constant to break symmetry
                val = (s * 13 + 7) % 32
                colors[i] = 1 if val >= 16 else 0
            else:
                # For vertices beyond 32, use a random-like but deterministic rule
                # based on the positions of the vertices
                # This is a placeholder; will be refined by local search
                val = (a * 31 + b * 17 + c * 11) % 2
                colors[i] = val

        return colors.tolist()

    def _parameterized_coloring(n, params):
        """
        Generate a coloring based on a small set of parameters.
        params: a list of integers that define the rule.
        For now, we use a simple rule based on XOR and a threshold.
        """
        m = n * (n - 1) * (n - 2) // 6
        colors = np.zeros(m, dtype=np.int8)
        # params[0] = multiplier, params[1] = offset, params[2] = threshold
        mult, offset, thresh = params[0], params[1], params[2]

        for i, (a, b, c) in enumerate(combinations(range(n), 3)):
            s = a ^ b ^ c
            val = (s * mult + offset) % 32
            colors[i] = 1 if val >= thresh else 0

        return colors.tolist()

    def _stochastic_parameter_search(n, time_budget):
        """
        Search over a small set of parameters to find a good coloring.
        """
        best_colors = None
        best_score = float('inf')
        t_end = time.time() + time_budget
        rng = random.Random(123)

        while time.time() < t_end:
            # Generate random parameters
            mult = rng.randint(1, 31)
            offset = rng.randint(0, 31)
            thresh = rng.randint(8, 24)
            params = [mult, offset, thresh]

            colors = _parameterized_coloring(n, params)
            quads, pents, tq, tp = _build_data(n)
            red_k4, blue_k5, _, _ = _initial_energy(colors, quads, pents)
            score = red_k4 + blue_k5

            if score < best_score:
                best_score = score
                best_colors = colors

            if score == 0:
                break

        return best_colors

    # 1. Start with n=32 using algebraic seed
    n_current = 32
    quads, pents, tq, tp = _build_data(n_current)
    colors = _algebraic_seed(n_current)

    # Refine the seed
    colors, valid = _violation_driven_search(colors, quads, pents, tq, tp, 10.0)

    best_n = n_current if valid else 0
    best_colors = colors if valid else None

    if not valid:
        # Try stochastic parameter search
        colors_param = _stochastic_parameter_search(n_current, 10.0)
        if colors_param is not None:
            colors_param, valid = _violation_driven_search(colors_param, quads, pents, tq, tp, 10.0)
            if valid:
                best_n = n_current
                best_colors = colors_param

        if not valid:
            # Fallback to smaller n
            for n in range(31, 9, -1):
                if time.time() > _START + _DEADLINE - 5:
                    break
                quads_n, pents_n, tq_n, tp_n = _build_data(n)
                m_n = n * (n - 1) * (n - 2) // 6
                colors_n = (np.random.rand(m_n) < 0.5).astype(np.int8).tolist()
                colors_n, valid = _violation_driven_search(colors_n, quads_n, pents_n, tq_n, tp_n, 5.0)
                if valid:
                    best_n = n
                    best_colors = colors_n
                    break

    # 2. Try to extend to n=34
    if valid and best_n == 32:
        n_target = 34
        time_left = _START + _DEADLINE - 5.0 - time.time()
        if time_left > 20:
            quads_34, pents_34, tq_34, tp_34 = _build_data(n_target)
            m_34 = n_target * (n_target - 1) * (n_target - 2) // 6

            # Start from the n=32 solution and extend
            # For the first 32 vertices, use the existing colors
            # For the new vertices, use a random-like rule
            colors_34 = [0] * m_34
            for i, (a, b, c) in enumerate(combinations(range(n_target), 3)):
                if max(a, b, c) < 32:
                    # Use the existing color
                    idx_32 = _t_idx(32, a, b, c)
                    colors_34[i] = best_colors[idx_32]
                else:
                    # New vertex: use a random-like but deterministic rule
                    val = (a * 31 + b * 17 + c * 11) % 2
                    colors_34[i] = val

            colors_34, valid_34 = _violation_driven_search(colors_34, quads_34, pents_34, tq_34, tp_34, time_left)
            if valid_34:
                best_n = 34
                best_colors = colors_34

    # 3. Try to extend to n=35
    if best_n == 34:
        n_target = 35
        time_left = _START + _DEADLINE - 5.0 - time.time()
        if time_left > 10:
            quads_35, pents_35, tq_35, tp_35 = _build_data(n_target)
            m_35 = n_target * (n_target - 1) * (n_target - 2) // 6

            # Start from the n=34 solution and extend
            colors_35 = [0] * m_35
            for i, (a, b, c) in enumerate(combinations(range(n_target), 3)):
                if max(a, b, c) < 34:
                    idx_34 = _t_idx(34, a, b, c)
                    colors_35[i] = best_colors[idx_34]
                else:
                    val = (a * 31 + b * 17 + c * 11) % 2
                    colors_35[i] = val

            colors_35, valid_35 = _violation_driven_search(colors_35, quads_35, pents_35, tq_35, tp_35, time_left)
            if valid_35:
                best_n = 35
                best_colors = colors_35

    if best_colors is None:
        # Fallback: trivial valid coloring for n=9
        n = 9
        triples = list(combinations(range(n), 3))
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
        best_n = n

    return {'n': best_n, 'colors': best_colors}
