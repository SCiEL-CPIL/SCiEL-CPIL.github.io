            delta = new_cost - current_cost
            if delta <= 0:
                accept = True
            else:
                if delta > temp * 10:
                    accept = False
                else:
                    accept = random.random() < np.exp(-delta / temp)
            if accept:
                network = new_network
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_network = network[:]
