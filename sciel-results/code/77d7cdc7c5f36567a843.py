<<<<<<< SEARCH
    # Genetic algorithm loop
    generation = 0
    no_improve = 0
    elite_count = 5

    while time_left() > 15:
        generation += 1
        if generation % 10 == 0:
            print(f"Gen {generation}, best={best_score}", file=sys.stderr)

        new_population = []

        # Elitism: keep top individuals
        for i in range(min(elite_count, len(population))):
            new_population.append(population[i])

        # Generate offspring
        while len(new_population) < pop_size:
            p1 = tournament_select(population, k=3)
            p2 = tournament_select(population, k=3)

            if random.random() < 0.3:
                # Crossover
                child = crossover(p1, p2)
            else:
                # Mutation of best parent
                child = mutate(p1, rate=0.1)

            # Local search on child (limited iterations)
            if time_left() > 20:
                child = local_search(child, max_iter=50)

            new_population.append(child)

        # Update population
        population = new_population
        population.sort(key=lambda x: x[2], reverse=True)

        # Track best
        if population[0][2] > best_score:
            best_score = population[0][2]
            best_list = population[0][0][:]
            best_mask = population[0][1]
            no_improve = 0
        else:
            no_improve += 1

        # Restart mechanism if stuck
        if no_improve > 30:
            # Add fresh random individuals to maintain diversity
            for _ in range(5):
                lst = random.sample(range(1, MAX_VAL + 1), N)
                m = list_to_mask(lst)
                s = score_mask(m)
                population.append((lst, m, s))
            # Also try structured restarts
            for mult in [2, 3]:
                sizeA = random.randint(3, 7)
                sizeB = random.randint(3, 7)
                A = set(random.sample(range(1, 60), sizeA))
                maxB = (MAX_VAL - 1) // mult
                B = set(random.sample(range(1, maxB + 1), sizeB))
                S = set()
                for a in A:
                    for b in B:
                        v = a + mult * b
                        if 1 <= v <= MAX_VAL:
                            S.add(v)
                if len(S) >= N:
                    lst = random.sample(sorted(S), N)
                    m = list_to_mask(lst)
                    s = score_mask(m)
                    population.append((lst, m, s))
            population.sort(key=lambda x: x[2], reverse=True)
            population = population[:pop_size]
            no_improve = 0
=======
    # Genetic algorithm loop
    generation = 0
    no_improve = 0
    elite_count = 5

    while time_left() > 20:
        generation += 1
        if generation % 10 == 0:
            print(f"Gen {generation}, best={best_score}", file=sys.stderr)

        new_population = []

        # Elitism: keep top individuals
        for i in range(min(elite_count, len(population))):
            new_population.append(population[i])

        # Generate offspring
        while len(new_population) < pop_size:
            p1 = tournament_select(population, k=3)
            p2 = tournament_select(population, k=3)

            if random.random() < 0.3:
                # Crossover
                child = crossover(p1, p2)
            else:
                # Mutation of best parent
                child = mutate(p1, rate=0.1)

            # Local search on child (limited iterations)
            if time_left() > 25:
                child = local_search(child, max_iter=50)

            new_population.append(child)

        # Update population
        population = new_population
        population.sort(key=lambda x: x[2], reverse=True)

        # Track best
        if population[0][2] > best_score:
            best_score = population[0][2]
            best_list = population[0][0][:]
            best_mask = population[0][1]
            no_improve = 0
        else:
            no_improve += 1

        # Restart mechanism if stuck
        if no_improve > 30:
            # Add fresh random individuals to maintain diversity
            for _ in range(5):
                lst = random.sample(range(1, MAX_VAL + 1), N)
                m = list_to_mask(lst)
                s = score_mask(m)
                population.append((lst, m, s))
            # Also try structured restarts
            for mult in [2, 3]:
                sizeA = random.randint(3, 7)
                sizeB = random.randint(3, 7)
                A = set(random.sample(range(1, 60), sizeA))
                maxB = (MAX_VAL - 1) // mult
                B = set(random.sample(range(1, maxB + 1), sizeB))
                S = set()
                for a in A:
                    for b in B:
                        v = a + mult * b
                        if 1 <= v <= MAX_VAL:
                            S.add(v)
                if len(S) >= N:
                    lst = random.sample(sorted(S), N)
                    m = list_to_mask(lst)
                    s = score_mask(m)
                    population.append((lst, m, s))
            population.sort(key=lambda x: x[2], reverse=True)
            population = population[:pop_size]
            no_improve = 0
>>>>>>> REPLACE

