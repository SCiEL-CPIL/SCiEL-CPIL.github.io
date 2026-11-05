    for a in range(self.n):
        x_aug = self._augment_context(context, a)
        theta = self.V @ self.b  # <--- This is constant for all arms!
        mu = theta @ x_aug
        ...
    