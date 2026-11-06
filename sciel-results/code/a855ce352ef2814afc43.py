import time
import random
import sys
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def get_fallback(n):
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}

    best_result = get_fallback(5)

    def edge_index(i, j, n):
        """Compute edge index for (i,j) in lexicographic order for K_n."""
        if i > j:
            i, j = j, i
        return i * (n - 2) - i * (i - 1) // 2 + j - 1

    def build_cycles(n):
        """Build list of all C6 cycles as lists of edge indices."""
        cycles = []
        verts = list(range(n))
        for combo in itertools.combinations(verts, 6):
            v0 = combo[0]
            others = combo[1:]
            for perm in itertools.permutations(others):
                # Enforce canonical form: perm[0] < perm[-1] to avoid reverse duplicates
                if perm[0] > perm[-1]:
                    continue
                path = [v0] + list(perm)
                cycle_edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    cycle_edges.append(edge_index(u, v, n))
                cycles.append(cycle_edges)
        return cycles

    def is_mono_cycle(colors, ce):
        """Check if a cycle is monochromatic."""
        c0 = colors[ce[0]]
        for e in ce[1:]:
            if colors[e] != c0:
                return False
        return True

    def count_bad_c6(colors, cycles):
        """Count all monochromatic C6s."""
        total = 0
        for ce in cycles:
            if is_mono_cycle(colors, ce):
                total += 1
        return total

    def find_bad_cycles(colors, cycles):
        """Find all monochromatic C6s."""
        bad = []
        for ci, ce in enumerate(cycles):
            if is_mono_cycle(colors, ce):
                bad.append(ci)
        return bad

    def walksat_repair(colors, n, cycles, edge_to_cycles, time_budget):
        """WalkSAT-style: pick random bad C6, change one edge in it."""
        start = time.time()
        m = n * (n - 1) // 2
        bad_count = count_bad_c6(colors, cycles)
        
        if bad_count == 0:
            return True, 0
        
        best_bad = bad_count
        best_colors = colors[:]
        
        # Precompute for each cycle, the edges (for quick access)
        # Use a set-based approach for bad cycles
        bad_set = set(find_bad_cycles(colors, cycles))
        
        max_iter = 500000
        for i in range(max_iter):
            if time.time() > start + time_budget:
                break
            
            if not bad_set:
                return True, 0
            
            # Pick a random bad cycle
            ci = random.choice(list(bad_set))
            ce = cycles[ci]
            
            # Pick a random edge in this cycle
            e = random.choice(ce)
            c_old = colors[e]
            
            # Pick a new color different from c_old
            c_new = random.randint(0, 3)
            while c_new == c_old:
                c_new = random.randint(0, 3)
            
            # Apply the change
            colors[e] = c_new
            
            # Update bad cycle set incrementally
            # Only cycles containing edge e can change status
            for ci2 in edge_to_cycles[e]:
                ce2 = cycles[ci2]
                now_mono = is_mono_cycle(colors, ce2)
                if now_mono:
                    bad_set.add(ci2)
                elif ci2 in bad_set:
                    bad_set.discard(ci2)
            
            bad_count = len(bad_set)
            if bad_count < best_bad:
                best_bad = bad_count
                best_colors = colors[:]
            
            if bad_count == 0:
                return True, 0
        
        if best_bad == 0:
            return True, 0
        return False, best_bad

    def anneal_polish(colors, n, cycles, edge_to_cycles, time_budget):
        """Simulated annealing to polish the coloring."""
        start = time.time()
        m = n * (n - 1) // 2
        bad_count = count_bad_c6(colors, cycles)
        
        if bad_count == 0:
            return True, 0
        
        best_bad = bad_count
        best_colors = colors[:]
        
        T_start = 1.0
        T_end = 0.01
        max_iter = 300000
        
        for i in range(max_iter):
            if time.time() > start + time_budget:
                break
            
            # Pick random edge
            e = random.randint(0, m - 1)
            c_old = colors[e]
            
            # Pick new color
            c_new = random.randint(0, 3)
            while c_new == c_old:
                c_new = random.randint(0, 3)
            
            # Calculate delta
            delta = 0
            for ci in edge_to_cycles[e]:
                ce = cycles[ci]
                cnt_old = 0
                cnt_new = 0
                for ed in ce:
                    if ed == e:
                        continue
                    if colors[ed] == c_old:
                        cnt_old += 1
                    elif colors[ed] == c_new:
                        cnt_new += 1
                
                before_mono = 1 if cnt_old == 5 else 0
                after_mono = 1 if cnt_new == 5 else 0
                delta += (after_mono - before_mono)
            
            # Acceptance
            progress = i / max_iter
            T = T_start * (T_end / T_start) ** progress
            if delta <= 0:
                accept = True
            else:
                if delta / T > 50:
                    accept = False
                else:
                    accept = random.random() < math.exp(-delta / T)
            
            if accept:
                colors[e] = c_new
                bad_count += delta
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_colors = colors[:]
                    if best_bad == 0:
                        return True, 0
        
        if best_bad == 0:
            return True, 0
        return False, best_bad

    def search_n(n, time_budget, seed_offset=0):
        if time.time() > deadline - 5:
            return None

        m = n * (n - 1) // 2
        random.seed(42 + seed_offset)

        # Build cycles
        cycles = build_cycles(n)
        num_cycles = len(cycles)

        if num_cycles == 0:
            colors = [0] * m
            return {'n': n, 'colors': colors}

        # Build edge-to-cycles mapping
        edge_to_cycles = [[] for _ in range(m)]
        for ci, ce in enumerate(cycles):
            for e in ce:
                edge_to_cycles[e].append(ci)

        # Try multiple initialization strategies
        init_strategies = []
        
        # Strategy 1: Pure random
        init_strategies.append('random')
        
        # Strategy 2: Modular-based
        # Color by (j - i) mod 7, mapped to 4 colors
        # Residues 0-6: map to colors 0,1,2,3,0,1,2 (roughly balanced)
        init_strategies.append('mod7')
        
        # Strategy 3: Two-phase modular
        init_strategies.append('mod5')
        
        # Allocate time to each strategy
        num_strategies = len(init_strategies)
        time_per_strategy = time_budget / num_strategies
        
        for strat in init_strategies:
            if time.time() > deadline - 5:
                break
            
            # Initialize colors based on strategy
            if strat == 'random':
                colors = [random.randint(0, 3) for _ in range(m)]
            elif strat == 'mod7':
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        d = (j - i) % 7
                        # Map 7 residues to 4 colors: 0->0, 1->1, 2->2, 3->3, 4->0, 5->1, 6->2
                        c = d % 4
                        colors.append(c)
            elif strat == 'mod5':
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        d = (j - i) % 5
                        # Map 5 residues to 4 colors: 0->0, 1->1, 2->2, 3->3, 4->(random or 0)
                        if d < 4:
                            c = d
                        else:
                            c = random.randint(0, 3)
                        colors.append(c)
            
            bad_count = count_bad_c6(colors, cycles)
            
            if bad_count == 0:
                return {'n': n, 'colors': colors}
            
            # Phase 1: WalkSAT repair
            ws_budget = time_per_strategy * 0.6
            success, remaining_bad = walksat_repair(colors, n, cycles, edge_to_cycles, ws_budget)
            
            if success:
                return {'n': n, 'colors': colors}
            
            # Phase 2: Annealing polish
            an_budget = time_per_strategy * 0.4
            if time.time() > deadline - 5:
                break
            success, remaining_bad = anneal_polish(colors, n, cycles, edge_to_cycles, an_budget)
            
            if success:
                return {'n': n, 'colors': colors}
        
        return None

    # Try targets from high to low
    targets = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]

    # Allocate time: prioritize higher n
    time_alloc = {17: 30, 16: 25, 15: 20, 14: 15, 13: 12, 12: 10, 11: 8, 10: 6, 9: 4, 8: 4, 7: 3, 6: 3}

    for n in targets:
        if time.time() > deadline - 8:
            break

        tb = time_alloc.get(n, 5)
        # Try multiple restarts for harder n
        num_restarts = 3 if n >= 15 else 2
        restart_budget = tb / num_restarts

        for r in range(num_restarts):
            if time.time() > deadline - 8:
                break
            result = search_n(n, restart_budget, seed_offset=r * 1000 + n)
            if result:
                best_result = result
                # Found valid coloring for n, return immediately since we go high to low
                return best_result

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
