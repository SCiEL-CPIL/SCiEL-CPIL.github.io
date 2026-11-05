            if delta < 0:
                accept = True
            else:
                accept = False
                if T > 0:
                    # Avoid overflow
                    if -delta / T < 700: # exp(700) is max
                        p = math.exp(-delta / T)
                    else:
                        p = 0.0
                    accept = random.random() < p
