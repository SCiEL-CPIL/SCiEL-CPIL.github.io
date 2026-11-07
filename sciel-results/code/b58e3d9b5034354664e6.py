import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        D = self.dim
        budget = self.budget
        
        # Differential Evolution Parameters
        # Population size. For D=5, NP=10 is standard.
        NP = 10
        
        # Initial parameters
        F = 0.8
        CR = 0.5
        
        # JADE adaptation parameters
        # c1 = 0.1, c2 = 0.1
        c1 = 0.1
        c2 = 0.1
        # Moving averages
        F_ma = F
        CR_ma = CR
        
        # Initial population
        # Uniform in [-5, 5]
        pop = np.random.uniform(-5, 5, size=(NP, D))
        fits = np.array([func(x) for x in pop])
        
        calls_made = NP
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]
        
        # Track successful F and CR values for adaptation
        F_success = []
        CR_success = []
        
        while calls_made + NP <= budget:
            # Generate trial population
            trial_pop = np.zeros((NP, D))
            trial_fits = np.zeros(NP)
            
            # Lists to store F and CR used for successful trials
            gen_F_success = []
            gen_CR_success = []
            
            for i in range(NP):
                # Select 3 distinct random indices different from i
                # Using choice with replace=False
                r1, r2, r3 = np.random.choice(NP, 3, replace=False)
                while r1 == i:
                    r1 = np.random.choice(NP)
                while r2 == i or r2 == r1:
                    r2 = np.random.choice(NP)
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = np.random.choice(NP)
                    
                # DE/rand/1 mutation
                # v_i = x_r1 + F * (x_r2 - x_r3)
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                # Random crossover rate for this individual? 
                # JADE usually uses a fixed CR per generation or individual.
                # Let's use CR_ma for this generation.
                j_rand = np.random.randint(D)
                trial = np.zeros(D)
                for j in range(D):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                        
                # Boundary handling: Clip to [-5, 5]
                trial = np.clip(trial, -5, 5)
                
                trial_pop[i] = trial
                # Evaluate
                trial_fits[i] = func(trial)
                calls_made += 1
                
                # Selection
                if trial_fits[i] < fits[i]:
                    pop[i] = trial
                    fits[i] = trial_fits[i]
                    # Record success
                    gen_F_success.append(F)
                    gen_CR_success.append(CR)
                    
                    if trial_fits[i] < best_f:
                        best_f = trial_fits[i]
                        best_x = trial.copy()
                        
            # Adapt F and CR (JADE)
            if gen_F_success:
                # Update moving averages
                # F = F_ma + c1 * (mean(F_success) - F_ma)
                # CR = CR_ma + c2 * (mean(CR_success) - CR_ma)
                # Note: JADE typically samples F and CR from Cauchy distributions centered at F_ma and CR_ma.
                # Here we simplify by just updating the mean and using that, or sampling.
                # Let's sample from Cauchy to allow exploration.
                F_ma = F_ma + c1 * (np.mean(gen_F_success) - F_ma)
                CR_ma = CR_ma + c2 * (np.mean(gen_CR_success) - CR_ma)
                
                # Sample new F and CR for next generation
                # Cauchy distribution: F = F_ma + 1/Cauchy(0,1)
                # But we need to keep them in valid ranges [0.1, 1.0] and [0, 1]
                F = np.clip(F_ma + np.random.standard_cauchy(), 0.1, 1.0)
                CR = np.clip(CR_ma + np.random.standard_cauchy(), 0.0, 1.0)
            else:
                # No improvement, keep current or decay?
                # JADE keeps them if no success, or uses previous.
                pass
                
        # If we have remaining budget, we could do a local search or just stop.
        # DE is stochastic, stopping is fine.
        
        return best_x, best_f
