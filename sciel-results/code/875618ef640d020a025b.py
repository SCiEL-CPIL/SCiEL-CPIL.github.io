import itertools
import time
import sys
import random
import math

def construct(n):
    # Deadline guard
    start_time = time.time()
    timeout = 115.0  # 5 sec margin

    # Classic construction
    def classic_superpermutation(n):
        if n == 1:
            return [1]
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find max overlap
                max_ov = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        max_ov = o
                        break
                new += block[max_ov:]
            word = new
        return word

    # Get classic result
    classic_word = classic_superpermutation(n)
    classic_len = len(classic_word)

    # For small n, classic is optimal
    if n <= 5:
        return "".join(map(str, classic_word))

    # Try Genetic Algorithm + Simulated Annealing for n=6, 7
    if n in [6, 7]:
        try:
            # Generate all permutations
            perms = list(itertools.permutations(range(1, n + 1)))
            N = len(perms)

            # Precompute overlap matrix: overlap[i][j] = max k such that
            # suffix of perms[i] of length k == prefix of perms[j] of length k
            overlap = [[0] * N for _ in range(N)]

            # Precompute prefix maps for each length
            prefix_maps = [None] * (n + 1)
            for k in range(1, n):
                pm = {}
                for i, p in enumerate(perms):
                    pref = p[:k]
                    if pref not in pm:
                        pm[pref] = []
                    pm[pref].append(i)
                prefix_maps[k] = pm

            # Compute overlap matrix using prefix maps
            for i in range(N):
                curr = perms[i]
                for k in range(n - 1, 0, -1):
                    suffix = curr[n - k:]
                    candidates = prefix_maps[k].get(suffix, [])
                    for j in candidates:
                        if overlap[i][j] == 0:
                            overlap[i][j] = k

            # Energy function: total length = N*n - sum of overlaps
            def compute_energy(order):
                total = n
                for i in range(N - 1):
                    total += n - overlap[order[i]][order[i + 1]]
                return total

            # Greedy ordering
            def greedy_order():
                visited = [False] * N
                order = [0]
                visited[0] = True
                current = 0
                for _ in range(N - 1):
                    best_j = -1
                    best_ov = 0
                    for k in range(n - 1, 0, -1):
                        suffix = perms[current][n - k:]
                        candidates = prefix_maps[k].get(suffix, [])
                        for j in candidates:
                            if not visited[j]:
                                best_j = j
                                best_ov = k
                                break
                        if best_j != -1:
                            break
                    if best_j == -1:
                        for j in range(N):
                            if not visited[j]:
                                best_j = j
                                best_ov = 0
                                break
                    visited[best_j] = True
                    order.append(best_j)
                    current = best_j
                return order

            # Build word from order
            def build_word(order):
                word = list(perms[order[0]])
                for i in range(N - 1):
                    ov = overlap[order[i]][order[i + 1]]
                    word.extend(perms[order[i + 1]][ov:])
                return word

            # Genetic Algorithm
            def run_ga(pop_size, generations, elite_ratio=0.1, mutation_rate=0.05, crossover_rate=0.7):
                random.seed(42)
                
                # Initialize population with diverse orderings
                population = []
                # Add greedy ordering
                population.append(greedy_order())
                # Add random orderings
                for _ in range(pop_size - 1):
                    order = list(range(N))
                    random.shuffle(order)
                    population.append(order)
                
                # Compute initial fitness
                fitnesses = [compute_energy(ind) for ind in population]
                best_idx = fitnesses.index(min(fitnesses))
                best_order = population[best_idx][:]
                best_energy = fitnesses[best_idx]
                
                elite_count = max(1, int(pop_size * elite_ratio))
                
                for gen in range(generations):
                    if time.time() - start_time > timeout - 10.0:
                        break
                    
                    # Sort by fitness
                    indexed_pop = list(zip(fitnesses, population))
                    indexed_pop.sort(key=lambda x: x[0])
                    population = [x[1] for x in indexed_pop]
                    fitnesses = [x[0] for x in indexed_pop]
                    
                    # Update best
                    if fitnesses[0] < best_energy:
                        best_energy = fitnesses[0]
                        best_order = population[0][:]
                    
                    # Create next generation
                    next_population = []
                    
                    # Elitism: keep best individuals
                    for i in range(elite_count):
                        next_population.append(population[i][:])
                    
                    # Fill rest with offspring
                    while len(next_population) < pop_size:
                        # Selection: tournament selection
                        def tournament_select():
                            candidates = random.sample(range(pop_size), 3)
                            best_cand = min(candidates, key=lambda i: fitnesses[i])
                            return population[best_cand][:]
                        
                        parent1 = tournament_select()
                        parent2 = tournament_select()
                        
                        # Crossover: order crossover
                        if random.random() < crossover_rate and len(parent1) > 1:
                            # Order crossover
                            a, b = sorted(random.sample(range(N), 2))
                            child = [None] * N
                            # Copy segment from parent1
                            for i in range(a, b + 1):
                                child[i] = parent1[i]
                            # Fill rest from parent2 in order
                            fill = [x for x in parent2 if x not in child[a:b+1]]
                            fill_idx = 0
                            for i in range(N):
                                if child[i] is None:
                                    child[i] = fill[fill_idx]
                                    fill_idx += 1
                        else:
                            child = parent1[:]
                        
                        # Mutation: swap mutation
                        for i in range(N):
                            if random.random() < mutation_rate:
                                j = random.randint(0, N - 1)
                                child[i], child[j] = child[j], child[i]
                        
                        next_population.append(child)
                    
                    population = next_population
                    fitnesses = [compute_energy(ind) for ind in population]
                
                return best_order, best_energy

            # Simulated Annealing with better move operators
            def run_sa():
                random.seed(43)
                current_order = greedy_order()
                current_energy = compute_energy(current_order)
                best_order = current_order[:]
                best_energy = current_energy
                
                T_start = 5.0
                T_end = 0.001
                alpha = 0.9995
                T = T_start
                
                # Precompute for faster delta computation
                while time.time() - start_time < timeout - 5.0:
                    # Random move: swap, reverse, or relocate
                    move_type = random.random()
                    
                    if move_type < 0.4:
                        # Swap two positions
                        i = random.randint(0, N - 1)
                        j = random.randint(0, N - 1)
                        if i == j:
                            continue
                        if abs(i - j) == 1:
                            continue  # Skip adjacent swaps (handled by relocate)
                        
                        old_a = current_order[i]
                        old_b = current_order[j]
                        
                        delta = 0
                        if i > 0:
                            delta += (n - overlap[current_order[i-1]][old_a])
                        if i < N - 1:
                            delta += (n - overlap[old_a][current_order[i+1]])
                        if j > 0:
                            delta += (n - overlap[current_order[j-1]][old_b])
                        if j < N - 1:
                            delta += (n - overlap[old_b][current_order[j+1]])
                        
                        current_order[i], current_order[j] = old_b, old_a
                        
                        if i > 0:
                            delta -= (n - overlap[current_order[i-1]][old_b])
                        if i < N - 1:
                            delta -= (n - overlap[old_b][current_order[i+1]])
                        if j > 0:
                            delta -= (n - overlap[current_order[j-1]][old_a])
                        if j < N - 1:
                            delta -= (n - overlap[old_a][current_order[j+1]])
                        
                        new_energy = current_energy + delta
                        
                        if new_energy < current_energy or (T > 0 and random.random() < math.exp(-delta / T)):
                            current_energy = new_energy
                            if new_energy < best_energy:
                                best_energy = new_energy
                                best_order = current_order[:]
                        else:
                            current_order[i], current_order[j] = old_a, old_b
                        
                    elif move_type < 0.7:
                        # Reverse a segment
                        i = random.randint(0, N - 2)
                        j = random.randint(i + 1, N - 1)
                        if j - i < 2:
                            continue
                        
                        # Compute delta for reversal
                        delta = 0
                        # Edges affected: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
                        # Old edges: (i-1,i), (i,i+1), ..., (j-1,j), (j,j+1)
                        # New edges: (i-1,j), (j,j-1), ..., (i+1,i), (i,i-1) -> wait, reversal changes order
                        # After reversal, position i has old_order[j], position j has old_order[i]
                        # Edges: (i-1, new_i), (new_i, new_{i+1}), ..., (new_{j-1}, new_j), (new_j, j+1)
                        # new_i = old_order[j], new_{i+1} = old_order[j-1], ..., new_j = old_order[i]
                        
                        if i > 0:
                            delta += (n - overlap[current_order[i-1]][current_order[i]])
                        if j < N - 1:
                            delta += (n - overlap[current_order[j]][current_order[j+1]])
                        
                        # Internal edges: (i, i+1), ..., (j-1, j)
                        for k in range(i, j):
                            delta += (n - overlap[current_order[k]][current_order[k+1]])
                        
                        # New edges
                        if i > 0:
                            delta -= (n - overlap[current_order[i-1]][current_order[j]])
                        if j < N - 1:
                            delta -= (n - overlap[current_order[i]][current_order[j+1]])
                        
                        # New internal edges: (j, j-1), (j-1, j-2), ..., (i+1, i)
                        for k in range(j, i, -1):
                            delta -= (n - overlap[current_order[k]][current_order[k-1]])
                        
                        # Perform reversal
                        current_order[i:j+1] = current_order[i:j+1][::-1]
                        
                        new_energy = current_energy + delta
                        
                        if new_energy < current_energy or (T > 0 and random.random() < math.exp(-delta / T)):
                            current_energy = new_energy
                            if new_energy < best_energy:
                                best_energy = new_energy
                                best_order = current_order[:]
                        else:
                            current_order[i:j+1] = current_order[i:j+1][::-1]
                            # Undo reversal
                        # Note: reversal is its own inverse, so undo is just reverse again
                        
                    else:
                        # Relocate: move one element to another position
                        i = random.randint(0, N - 1)
                        j = random.randint(0, N -  1)
                        if i == j or abs(i - j) == 1:
                            continue
                        
                        elem = current_order[i]
                        
                        # Remove from i
                        delta = 0
                        if i > 0:
                            delta += (n - overlap[current_order[i-1]][elem])
                        if i < N - 1:
                            delta += (n - overlap[elem][current_order[i+1]])
                        if i > 0 and i < N - 1:
                            delta += (n - overlap[current_order[i-1]][current_order[i+1]])
                        
                        # Adjust j if j > i
                        if j > i:
                            j -= 1
                        
                        # Insert at j
                        if j > 0:
                            delta += (n - overlap[current_order[j-1]][elem])
                        if j < N - 1:
                            delta += (n - overlap[elem][current_order[j]])
                        if j > 0 and j < N - 1:
                            delta -= (n - overlap[current_order[j-1]][current_order[j]])
                        
                        # Perform move
                        current_order.pop(i)
                        current_order.insert(j, elem)
                        
                        new_energy = current_energy + delta
                        
                        if new_energy < current_energy or (T > 0 and random.random() < math.exp(-delta / T)):
                            current_energy = new_energy
                            if new_energy < best_energy:
                                best_energy = new_energy
                                best_order = current_order[:]
                        else:
                            # Undo move
                            current_order.pop(j)
                            current_order.insert(i, elem)
                    
                    T *= alpha
                    if T < T_end:
                        T = T_end
                
                return best_order, best_energy

            # Run GA first
            ga_order, ga_energy = run_ga(pop_size=50, generations=200)
            
            # Run SA starting from GA result
            random.seed(43)
            sa_start_order = ga_order[:]
            sa_order, sa_energy = run_sa()
            
            # Build best word
            best_order = sa_order if sa_energy < ga_energy else ga_order
            best_energy = min(sa_energy, ga_energy)
            
            best_word = build_word(best_order)
            best_len = len(best_word)

            if best_len < classic_len:
                return "".join(map(str, best_word))

            return "".join(map(str, classic_word))

        except Exception:
            pass

    return "".join(map(str, classic_word))
