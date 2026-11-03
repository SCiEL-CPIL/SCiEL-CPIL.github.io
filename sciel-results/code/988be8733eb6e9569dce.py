            if time.time() < deadline - 10:
                # perturbed = local_search(perturbed, deadline - 5)
                # perturbed = reduce_code(perturbed)
                perturbed = simulated_annealing(perturbed, deadline - 5)
                perturbed = reduce_code(perturbed)
