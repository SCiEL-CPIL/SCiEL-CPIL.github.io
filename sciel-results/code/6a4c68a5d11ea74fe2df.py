<<<<<<< SEARCH
                if new_energy <= current_energy or (T > min_t and rng.random() < math.exp(-new_energy / T)):
=======
                accept = False
                if new_energy <= current_energy:
                    accept = True
                elif T > min_t:
                    exp_arg = -new_energy / T
                    if exp_arg > 700:
                        accept = True
                    elif exp_arg > -700:
                        accept = rng.random() < math.exp(exp_arg)
                    else:
                        accept = False
                if accept:
>>>>>>> REPLACE
