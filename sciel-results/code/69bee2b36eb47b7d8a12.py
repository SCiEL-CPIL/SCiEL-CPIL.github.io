    def evaluate(P):
        """
        Compute the score H_D / M.
        """
        Hs, _, _, M, _ = compute_entropies_and_grads(P)
        if M < 1e-9:
            return -1.0
        return Hs['D'] / M
