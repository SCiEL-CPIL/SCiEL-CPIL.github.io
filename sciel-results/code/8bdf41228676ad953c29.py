"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from itertools import combinations
from collections import defaultdict
import random

# EVOLVE-BLOCK-START

def _triple_index(a, b, c, n):
    """Compute the lexicographic index of triple (a,b,c) where a<b<c among all triples of n vertices."""
    # Number of triples with first element < a
    # For first element i < a: C(n-1-i, 2)
    # For first element a, second element j < b (j > a): C(n-1-j, 1) = n-1-j
    # For first element a, second element b, third element < c (c > b): c - b - 1
    idx = 0
    for i in range(a):
        idx += (n - 1 - i) * (n - 2 - i) // 2
    for j in range(a + 1, b):
        idx += n - 1 - j
    idx += c - b - 1
    return idx


def _check_validity(n, colors):
    """Check that no 4-set is all-red and no 5-set is all-blue."""
    # Check red K4: every 4-set must have at least one blue triple
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        if (colors[_triple_index(a, b, c, n)] == 0 and
            colors[_triple_index(a, b, d, n)] == 0 and
            colors[_triple_index(a, c, d, n)] == 0 and
            colors[_triple_index(b, c, d, n)] == 0):
            return False
    
    # Check blue K5: every 5-set must have at least one red triple
    for quint in combinations(range(n), 5):
        a, b, c, d, e = quint
        triples = [
            (a, b, c), (a, b, d), (a, b, e),
            (a, c, d), (a, c, e), (a, d, e),
            (b, c, d), (b, c, e), (b, d, e),
            (c, d, e)
        ]
        if all(colors[_triple_index(x, y, z, n)] == 1 for x, y, z in triples):
            return False
    
    return True


def _count_violations_fast(n, colors):
    """Count the number of red K4 violations and blue K5 violations."""
    red_viol = 0
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        if (colors[_triple_index(a, b, c, n)] == 0 and
            colors[_triple_index(a, b, d, n)] == 0 and
            colors[_triple_index(a, c, d, n)] == 0 and
            colors[_triple_index(b, c, d, n)] == 0):
            red_viol += 1
    
    blue_viol = 0
    for quint in combinations(range(n), 5):
        a, b, c, d, e = quint
        triples = [
            (a, b, c), (a, b, d), (a, b, e),
            (a, c, d), (a, c, e), (a, d, e),
            (b, c, d), (b, c, e), (b, d, e),
            (c, d, e)
        ]
        if all(colors[_triple_index(x, y, z, n)] == 1 for x, y, z in triples):
            blue_viol += 1
    
    return red_viol + blue_viol


def _local_search(n, max_iters, deadline, seed=42):
    """Local search to find a valid 2-coloring of the 3-uniform hypergraph on n vertices."""
    random.seed(seed)
    
    num_triples = n * (n - 1) * (n - 2) // 6
    colors = [random.randint(0, 1) for _ in range(num_triples)]
    
    # Precompute for each triple, which quads and quints it belongs to
    # This is expensive for large n, so we'll use a simpler approach
    
    # For efficiency, let's track violations incrementally
    # But for simplicity, let's just do full recounts for small n
    
    def count_violations(c):
        red_viol = 0
        for quad in combinations(range(n), 4):
            a, b, c_, d = quad
            if (c[_triple_index(a, b, c_, n)] == 0 and
                c[_triple_index(a, b, d, n)] == 0 and
                c[_triple_index(a, c_, d, n)] == 0 and
                c[_triple_index(b, c_, d, n)] == 0):
                red_viol += 1
        blue_viol = 0
        for quint in combinations(range(n), 5):
            a, b, c_, d, e = quint
            triples = [
                (a, b, c_), (a, b, d), (a, b, e),
                (a, c_, d), (a, c_, e), (a, d, e),
                (b, c_, d), (b, c_, e), (b, d, e),
                (c_, d, e)
            ]
            if all(c[_triple_index(x, y, z, n)] == 1 for x, y, z in triples):
                blue_viol += 1
        return red_viol + blue_viol
    
    cur_viol = count_violations(colors)
    best_colors = colors[:]
    best_viol = cur_viol
    
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        if cur_viol == 0:
            break
        
        # Pick a random triple to flip
        idx = random.randint(0, num_triples - 1)
        old_color = colors[idx]
        colors[idx] = 1 - old_color
        
        new_viol = count_violations(colors)
        delta = new_viol - cur_viol
        if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
            cur_viol = new_viol
            if cur_viol < best_viol:
                best_viol = cur_viol
                best_colors = colors[:]
        else:
            colors[idx] = old_color
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    if best_viol == 0:
        return best_colors
    return None


def _algebraic_construction(n):
    """Try an algebraic construction based on finite fields or groups."""
    # Try using Z_3^k constructions
    # For n up to 34, we can try various group structures
    
    # One approach: vertices as elements of Z_m, color based on sum mod k
    # Let's try: color triple (a,b,c) red if (a+b+c) mod 3 == 0
    
    # Actually, let's try a more sophisticated approach
    # Use the fact that for R(4,5;3) > 34, there's a known construction
    
    # Try: vertices 0..n-1, color based on (a+b+c) mod 5
    # Red if (a+b+c) mod 5 in {0, 1}, Blue otherwise
    
    # Let's try several algebraic rules and pick the best
    best_colors = None
    best_viol = float('inf')
    
    for mod in [3, 4, 5, 6, 7]:
        for red_set in range(1 << mod):
            if red_set == 0 or red_set == (1 << mod) - 1:
                continue
            colors = []
            for a, b, c in combinations(range(n), 3):
                s = (a + b + c) % mod
                if red_set & (1 << s):
                    colors.append(0)
                else:
                    colors.append(1)
            
            # Quick check: count violations
            red_viol = 0
            for quad in combinations(range(n), 4):
                a, b, c_, d = quad
                if (colors[_triple_index(a, b, c_, n)] == 0 and
                    colors[_triple_index(a, b, d, n)] == 0 and
                    colors[_triple_index(a, c_, d, n)] == 0 and
                    colors[_triple_index(b, c_, d, n)] == 0):
                    red_viol += 1
                if red_viol > best_viol:
                    break
            
            if red_viol > best_viol:
                continue
            
            blue_viol = 0
            for quint in combinations(range(n), 5):
                a, b, c_, d, e = quint
                triples = [
                    (a, b, c_), (a, b, d), (a, b, e),
                    (a, c_, d), (a, c_, e), (a, d, e),
                    (b, c_, d), (b, c_, e), (b, d, e),
                    (c_, d, e)
                ]
                if all(colors[_triple_index(x, y, z, n)] == 1 for x, y, z in triples):
                    blue_viol += 1
            
            total = red_viol + blue_viol
            if total < best_viol:
                best_viol = total
                best_colors = colors[:]
    
    if best_viol == 0:
        return best_colors
    return None


def _greedy_construction(n, deadline):
    """Greedy construction: assign colors one triple at a time to avoid creating violations."""
    num_triples = n * (n - 1) * (n - 2) // 6
    colors = [-1] * num_triples
    
    # Precompute all quads and quints
    quads = list(combinations(range(n), 4))
    quints = list(combinations(range(n), 5))
    
    def get_triple_indices_for_quad(quad):
        a, b, c, d = quad
        return [
            _triple_index(a, b, c, n),
            _triple_index(a, b, d, n),
            _triple_index(a, c, d, n),
            _triple_index(b, c, d, n)
        ]
    
    def get_triple_indices_for_quint(quint):
        a, b, c, d, e = quint
        return [
            _triple_index(a, b, c, n),
            _triple_index(a, b, d, n),
            _triple_index(a, b, e, n),
            _triple_index(a, c, d, n),
            _triple_index(a, c, e, n),
            _triple_index(a, d, e, n),
            _triple_index(b, c, d, n),
            _triple_index(b, c, e, n),
            _triple_index(b, d, e, n),
            _triple_index(c, d, e, n)
        ]
    
    quad_indices = [get_triple_indices_for_quad(q) for q in quads]
    quint_indices = [get_triple_indices_for_quint(q) for q in quints]
    
    # For each triple, which quads and quints does it belong to?
    triple_to_quads = defaultdict(list)
    triple_to_quints = defaultdict(list)
    for i, qt in enumerate(quad_indices):
        for t in qt:
            triple_to_quads[t].append(i)
    for i, qt in enumerate(quint_indices):
        for t in qt:
            triple_to_quints[t].append(i)
    
    # Greedy: assign colors in lexicographic order
    # For each triple, try red first, then blue, pick the one that doesn't create a violation
    # If both create violations, pick the one that creates fewer future violations
    
    for idx in range(num_triples):
        if time.time() > deadline:
            break
        
        # Try red (0)
        can_red = True
        # Check all quads containing this triple
        for qi in triple_to_quads[idx]:
            qt = quad_indices[qi]
            # Check if all other triples in this quad are already red
            all_others_red = True
            for t in qt:
                if t != idx:
                    if colors[t] != 0:
                        all_others_red = False
                        break
            if all_others_red:
                can_red = False
                break
        
        # Try blue (1)
        can_blue = True
        # Check all quints containing this triple
        for qi in triple_to_quints[idx]:
            qt = quint_indices[qi]
            # Check if all other triples in this quint are already blue
            all_others_blue = True
            for t in qt:
                if t != idx:
                    if colors[t] != 1:
                        all_others_blue = False
                        break
            if all_others_blue:
                can_blue = False
                break
        
        if can_red and not can_blue:
            colors[idx] = 0
        elif can_blue and not can_red:
            colors[idx] = 1
        elif can_red and can_blue:
            # Prefer the color that leaves more options
            # Count how many quads/quints would be "blocked"
            red_blocks = 0
            for qi in triple_to_quads[idx]:
                qt = quad_indices[qi]
                # If we color this red, does it make this quad all-red?
                all_others_red = True
                for t in qt:
                    if t != idx:
                        if colors[t] != 0:
                            all_others_red = False
                            break
                if all_others_red:
                    red_blocks += 1
            
            blue_blocks = 0
            for qi in triple_to_quints[idx]:
                qt = quint_indices[qi]
                all_others_blue = True
                for t in qt:
                    if t != idx:
                        if colors[t] != 1:
                            all_others_blue = False
                            break
                if all_others_blue:
                    blue_blocks += 1
            
            if red_blocks < blue_blocks:
                colors[idx] = 0
            elif blue_blocks < red_blocks:
                colors[idx] = 1
            else:
                colors[idx] = random.randint(0, 1)
        else:
            # Neither works - this is a dead end
            # Try to fix by changing some previous colors
            # For simplicity, just pick randomly
            colors[idx] = random.randint(0, 1)
    
    # Check if the result is valid
    if all(c >= 0 for c in colors):
        # Verify
        red_viol = 0
        for qi, qt in enumerate(quad_indices):
            if all(colors[t] == 0 for t in qt):
                red_viol += 1
        blue_viol = 0
        for qi, qt in enumerate(quint_indices):
            if all(colors[t] == 1 for t in qt):
                blue_viol += 1
        
        if red_viol == 0 and blue_viol == 0:
            return colors
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with the seed answer: n=9
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Try algebraic constructions for various n
    for n in range(10, 35):
        if time.time() > deadline - 10:
            break
        
        colors = _algebraic_construction(n)
        if colors is not None:
            if _check_validity(n, colors):
                best_n = n
                best_colors = colors
    
    # Try greedy construction
    for n in range(best_n + 1, 35):
        if time.time() > deadline - 20:
            break
        
        for seed in [42, 123, 999, 777, 555, 333]:
            if time.time() > deadline - 5:
                break
            random.seed(seed)
            colors = _greedy_construction(n, deadline - 5)
            if colors is not None:
                if _check_validity(n, colors):
                    best_n = n
                    best_colors = colors
                    break
    
    # Try local search for the best n we found and slightly above
    for n in range(best_n, min(best_n + 5, 35)):
        if time.time() > deadline - 5:
            break
        
        for seed in [42, 123, 999, 777, 555, 333, 111, 222]:
            if time.time() > deadline - 2:
                break
            colors = _local_search(n, max_iters=50000, deadline=deadline - 2, seed=seed)
            if colors is not None:
                if _check_validity(n, colors):
                    best_n = n
                    best_colors = colors
                    break
    
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
