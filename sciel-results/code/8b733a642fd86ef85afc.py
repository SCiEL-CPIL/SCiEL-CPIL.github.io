   def construct():
       start_time = time.time()
       deadline = start_time + 140 # Safety margin
       
       vectors = generate_weight_4_vectors()
       N = len(vectors)
       if N == 0:
           return generate_weight_1_vectors()
       
       dots = vectors @ vectors.T
       conflicts = (dots > 2).astype(np.int8)
       np.fill_diagonal(conflicts, 0)
       
       rng = np.random.default_rng(42)
       
       # Phase 1: Greedy MIS restarts
       # Run several greedy restarts to find a good initial solution
       best_set = []
       mis_deadline = deadline - 60 # Leave 60s for ILS
       
       # Run at least 5 restarts, or until time
       n_restarts = 0
       while time.time() < mis_deadline:
           current_set = greedy_mis(conflicts, mis_deadline, rng)
           n_restarts += 1
           if len(current_set) > len(best_set):
               best_set = current_set
           
           # If we've had a decent result and time is running out for greedy, break
           # Greedy is fast, so we can run many.
       
       # Phase 2: Local Improvement on best_set
       if len(best_set) > 0 and time.time() < deadline - 10:
           improved_set = local_improve_fast(list(best_set), vectors, conflicts, deadline - 5, rng)
           if len(improved_set) > len(best_set):
               best_set = improved_set
       
       # Phase 3: Iterative Local Search (ILS)
       # Perturb and re-improve
       current_best = list(best_set)
       current_best_size = len(current_best)
       
       # Determine perturbation size
       # Removing too many loses too much; removing too few might not escape.
       # Try small perturbations.
       perturb_size = max(2, current_best_size // 10)
       
       while time.time() < deadline - 5:
           # Perturb
           perturbed = list(current_best)
           if len(perturbed) > perturb_size:
               # Remove random vertices
               remove_indices = rng.choice(len(perturbed), size=perturb_size, replace=False)
               # Sort to remove from end first to maintain list integrity easily?
               # Or just create a mask
               mask = np.ones(len(perturbed), dtype=bool)
               mask[remove_indices] = False
               perturbed = [perturbed[i] for i in range(len(perturbed)) if mask[i]]
           
           # Local Improve
           improved = local_improve_fast(perturbed, vectors, conflicts, deadline - 3, rng)
           
           if len(improved) > current_best_size:
               current_best = improved
               current_best_size = len(improved)
               # Optionally reduce perturbation size on success?
               # Or keep it constant.
       
       best_set = current_best
       
       if not best_set:
           return generate_weight_1_vectors()
       
       result_vectors_w4 = vectors[best_set].astype(np.int64)
       weight1_vectors = generate_weight_1_vectors()
       final_vectors = np.vstack([result_vectors_w4, weight1_vectors])
       
       return final_vectors
   