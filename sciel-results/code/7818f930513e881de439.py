            # Propose move: small step or jump
            if random.random() < 0.8:
                step = random.randint(-3, 3)
                new_x = old_x + step
                new_y = old_y + random.randint(-3, 3)
            else:
                new_x = random.randint(-20, 20)
                new_y = random.randint(-20, 20)
