import time
import random
import math

def construct():
    n = 256
    
    # Precompute APs
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
            
    num_aps = len(ap_list)

    # Precompute other_bits for fast delta calculation
    # other_bits[i] contains list of (o1, o2, o3) for each AP containing i
    other_bits = [[] for _ in range(n)]
    for ap in ap_list:
        a, b, c, d = ap
        other_bits[a].append((b, c, d))
        other_bits[b].append((a, c, d))
        other_bits[c].append((a, b, d))
        other_bits[d].append((a, b, c))

    def calculate_total_cost(colors):
        cnt = 0
        # Using local variables for speed
        cols = colors
        for ap in ap_list:
            if cols[ap[0]] == cols[ap[1]] == cols[ap[2]] == cols[ap[3]]:
                cnt += 1
        return cnt

    def get_cost_delta(colors, i):
        """
        Returns the change in number of mono APs if we flip bit i.
        """
        c_i = colors[i]
        new_c = 1 - c_i
        
        count_before = 0
        count_after = 0
        
        for o1, o2, o3 in other_bits[i]:
            c1 = colors[o1]
            c2 = colors[o2]
            c3 = colors[o3]
            
            if c1 == c2 == c3 == c_i:
                count_before += 1
                
            if c1 == c2 == c3 == new_c:
                count_after += 1
                
        return count_after - count_before

    def local_search_first_improvement(colors, max_steps=1000):
        """
        Performs local search by flipping bits that improve the cost.
        Uses first improvement strategy to be fast.
        """
        current_cost = calculate_total_cost(colors)
        steps = 0
        improved = True
        
        while improved and steps < max_steps:
            improved = False
            # Scan all bits to find an improving flip
            # To avoid bias, we could randomize the order, but sequential is faster
            for i in range(n):
                delta = get_cost_delta(colors, i)
                if delta < 0:
                    colors[i] = 1 - colors[i]
                    current_cost += delta
                    improved = True
                    steps += 1
                    # Restart scan from beginning to capture new opportunities
                    # or continue? First improvement usually continues or restarts.
                    # Restarting ensures we don't miss a better flip earlier in the array
                    # but continuing is faster. Let's try restarting for better quality.
                    break 
            if not improved:
                break
                
        return colors, current_cost

    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
        
    def random_init(n):
        return [random.randint(0, 1) for _ in range(n)]

    def greedy_init(n):
        # Try to build a coloring that avoids mono APs greedily?
        # Simple random is often better for GA diversity.
        # Let's use a pattern based on modulo
        return [(i // 4) % 2 for i in range(n)]

    def uniform_crossover(c1, c2):
        return [c1[i] if random.random() < 0.5 else c2[i] for i in range(n)]

    def mutate(c, rate=0.05):
        c = c[:]
        for i in range(n):
            if random.random() < rate:
                c[i] = 1 - c[i]
        return c

    def tournament_selection(pop, costs, k=3):
        selected = random.sample(range(len(pop)), k)
        best_idx = min(selected, key=lambda i: costs[i])
        return pop[best_idx][:]

    # Main GA Loop
    best_score = float('inf')
    best_colors = None
    
    deadline = time.time() + 110
    
    # Initialize Population
    pop_size = 30
    population = []
    costs = []
    
    # Diverse Initialization
    for i in range(pop_size):
        if i % 3 == 0:
            colors = thue_morse(n)
        elif i % 3 == 1:
            colors = random_init(n)
        else:
            colors = greedy_init(n)
            
        # Run local search to refine initial population
        colors, cost = local_search_first_improvement(colors, max_steps=200)
        population.append(colors)
        costs.append(cost)
        
        if cost < best_score:
            best_score = cost
            best_colors = colors[:]

    # GA Evolution
    generation = 0
    while time.time() < deadline - 5:
        generation += 1
        
        # Select Parents and Create Offspring
        offspring = []
        offspring_costs = []
        
        # Create half the population size of offspring
        for _ in range(pop_size // 2):
            parent1 = tournament_selection(population, costs)
            parent2 = tournament_selection(population, costs)
            
            child = uniform_crossover(parent1, parent2)
            child = mutate(child, rate=0.05)
            
            # Local Search on child
            child, cost = local_search_first_improvement(child, max_steps=100)
            
            offspring.append(child)
            offspring_costs.append(cost)
            
            if cost < best_score:
                best_score = cost
                best_colors = child[:]

        # Replace worst individuals in population with offspring
        # Sort population by cost
        combined_indices = sorted(range(len(population)), key=lambda i: costs[i])
        # Keep the best half
        keep_indices = combined_indices[:pop_size // 2]
        
        new_population = [population[i][:] for i in keep_indices]
        new_costs = [costs[i] for i in keep_indices]
        
        # Add offspring
        new_population.extend(offspring)
        new_costs.extend(offspring_costs)
        
        population = new_population
        costs = new_costs
        
        # Check for convergence or time
        if best_score == 0:
            break

    return best_colors
