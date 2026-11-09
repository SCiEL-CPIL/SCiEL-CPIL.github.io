        while eval_count < self.budget:
            # Check if we can fit a full generation?
            # Or just fill up to budget.
            
            fronts = fast_non_dominated_sort(objectives)
            all_indices = list(range(pop_size))
            cd_dict = crowding_distance(objectives, all_indices)
            
            offspring = []
            offspring_objs = []
            
            # We want to generate offspring until we hit the budget
            # But NSGA-II works on full populations. 
            # If budget is not enough for a full generation, we can generate partial offspring,
            # but then the "combined" population size is not 2*pop_size.
            # Given budget=200 and pop_size=40, 200 = 40 (init) + 40 (gen1) + 40 (gen2) + 40 (gen3) + 40 (gen4)?
            # 40 * 5 = 200.
            # So it fits perfectly if we do 4 generations of full offspring.
            
            # But to be safe for any budget:
            # Let's just run generations while we have budget for a full set of offspring?
            # Or adjust pop_size?
            # The contract fixes budget=200.
            
            # Let's stick to the previous logic but add a pre-check.
            
            for _ in range(pop_size):
                if eval_count + 2 > self.budget:
                    # Can't fit 2 more
                    # If we can fit 1, we could, but NSGA-II needs pairs?
                    # Actually, we can just break.
                    # If we break, we have partial offspring.
                    # We should probably just not start this generation if we can't fit it?
                    # Or handle the last partial generation.
                    # Given 200 and 40, it fits exactly.
                    pass
                
                # Selection...
                # Crossover...
                # Mutation...
                # Evaluate...
                
                eval_count += 2
                if eval_count > self.budget:
                    # This should not happen if we check properly
                    # But if it does, we must undo? No, we can't undo.
                    # So we MUST check before.
                    pass
