import time
import sys
import random
from itertools import combinations

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit
    random.seed(42)  # Deterministic for reproducibility
    
    # Base cycle edges: 0-1-2-...-39-0
    base_edges = [(i, (i + 1) % N) for i in range(N)]
    base_edge_set = set((min(e), max(e)) for e in base_edges)
    
    # We need to cover cycle lengths 3 through 39 (length 40 is the base cycle)
    target_lengths = set(range(3, N))  # 3 to 39
    
    # Generate all possible chords and their coverage
    chord_coverages = []
    chord_index = {}
    for u in range(N):
        for v in range(u + 1, N):
            if (u, v) in base_edge_set:
                continue
            # Compute the two cycle lengths this chord creates
            len1 = v - u + 1
            len2 = N - (v - u) + 1
            # Both lengths must be >= 3 to be useful
            covered = set()
            if 3 <= len1 <= 39:
                covered.add(len1)
            if 3 <= len2 <= 39:
                covered.add(len2)
            if covered:
                idx = len(chord_coverages)
                chord_coverages.append(((u, v), covered))
                chord_index[(u, v)] = idx
    
    num_chords = len(chord_coverages)
    
    def greedy_cover():
        """Greedy set cover to get a good initial solution."""
        uncovered = set(target_lengths)
        selected = set()
        
        while uncovered:
            best_idx = None
            best_count = 0
            for idx, (chord, covered) in enumerate(chord_coverages):
                if idx in selected:
                    continue
                new_cov = len(covered & uncovered)
                if new_cov > best_count:
                    best_count = new_cov
                    best_idx = idx
            
            if best_idx is None or best_count == 0:
                break
            selected.add(best_idx)
            uncovered -= chord_coverages[best_idx][1]
        
        return selected
    
    def count_edges(selected_chord_indices):
        """Count total edges given a set of chord indices."""
        return N + len(selected_chord_indices)
    
    def is_valid(selected_chord_indices):
        """Check if the selected chords cover all required lengths."""
        covered = set()
        for idx in selected_chord_indices:
            covered |= chord_coverages[idx][1]
        return covered == target_lengths
    
    # Get initial solution with greedy
    best_solution = greedy_cover()
    best_score = count_edges(best_solution)
    
    # Simulated annealing to improve
    current_solution = set(best_solution)
    current_score = best_score
    
    temp = 10.0
    cooling = 0.995
    min_temp = 0.1
    
    while temp > min_temp and (time.time() - start_time) < deadline:
        # Random move: add, remove, or swap a chord
        move_type = random.random()
        new_solution = set(current_solution)
        
        if move_type < 0.3 and len(new_solution) > 0:
            # Remove a chord
            to_remove = random.choice(list(new_solution))
            new_solution.discard(to_remove)
        elif move_type < 0.6:
            # Add a chord
            candidates = [i for i in range(num_chords) if i not in new_solution]
            if candidates:
                to_add = random.choice(candidates)
                new_solution.add(to_add)
        else:
            # Swap: remove one, add one
            if len(new_solution) > 0:
                to_remove = random.choice(list(new_solution))
                candidates = [i for i in range(num_chords) if i not in new_solution]
                if candidates:
                    to_add = random.choice(candidates)
                    new_solution.discard(to_remove)
                    new_solution.add(to_add)
        
        # Check validity
        if not is_valid(new_solution):
            # Invalid solution: only accept if temperature is high enough
            # and the solution is "close" to valid (penalty-based)
            covered = set()
            for idx in new_solution:
                covered |= chord_coverages[idx][1]
            missing = len(target_lengths - covered)
            penalty = missing * 10
            new_score = count_edges(new_solution) + penalty
            
            if new_score < current_score or random.random() < (1.0 / (1.0 + abs(new_score - current_score) / temp)):
                current_solution = new_solution
                current_score = count_edges(new_solution)
        else:
            new_score = count_edges(new_solution)
            
            if new_score < current_score:
                current_solution = new_solution
                current_score = new_score
                
                if new_score < best_score:
                    best_score = new_score
                    best_solution = set(current_solution)
            elif random.random() < (1.0 / (1.0 + abs(new_score - current_score) / temp)):
                current_solution = new_solution
                current_score = new_score
        
        temp *= cooling
    
    # Use the best solution found
    selected_indices = best_solution
    
    # Build final edge set
    all_edges = list(base_edges)
    for idx in selected_indices:
        u, v = chord_coverages[idx][0]
        edge = (min(u, v), max(u, v))
        if edge not in base_edge_set:
            all_edges.append(edge)
    
    # Build edge set for verification
    edge_set = set((min(e), max(e)) for e in all_edges)
    
    # Generate cycles for each length 3..40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Base cycle
            cycles.append(list(range(N)))
            continue
        
        # Find a chord that creates a cycle of length L
        found = False
        for u in range(N):
            for v in range(u + 1, N):
                len1 = v - u + 1
                len2 = N - (v - u) + 1
                if len1 == L or len2 == L:
                    edge = (min(u, v), max(u, v))
                    if edge in edge_set:
                        if len1 == L:
                            # Cycle: u, u+1, ..., v, u
                            cycle = list(range(u, v + 1))
                        else:
                            # Cycle: u, u-1, ..., v, u (going the long way)
                            cycle = [u]
                            cur = (u - 1) % N
                            while cur != v:
                                cycle.append(cur)
                                cur = (cur - 1) % N
                            cycle.append(v)
                        # Verify the cycle
                        valid = True
                        for i in range(len(cycle)):
                            a = cycle[i]
                            b = cycle[(i + 1) % len(cycle)]
                            if (min(a, b), max(a, b)) not in edge_set:
                                valid = False
                                break
                        if valid and len(set(cycle)) == L:
                            cycles.append(cycle)
                            found = True
                            break
            if found:
                break
        
        if not found:
            # Fallback: try all possible cycles of length L
            for u in range(N):
                cycle = [(u + i) % N for i in range(L)]
                valid = True
                for i in range(L):
                    a = cycle[i]
                    b = cycle[(i + 1) % L]
                    if (min(a, b), max(a, b)) not in edge_set:
                        valid = False
                        break
                if valid and len(set(cycle)) == L:
                    cycles.append(cycle)
                    found = True
                    break
            if not found:
                cycles.append(list(range(L)))
    
    # Final verification
    for cycle in cycles:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            assert (min(u, v), max(u, v)) in edge_set, f"Edge ({u},{v}) not in graph"
        assert len(set(cycle)) == len(cycle), f"Cycle has duplicate vertices: {cycle}"
    
    return {'edges': all_edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
